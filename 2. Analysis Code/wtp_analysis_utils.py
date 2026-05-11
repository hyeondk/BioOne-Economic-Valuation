## 본설문 WTP 로짓 분석 공통 함수 ##
# 노노트북에서는 P4(이용료 정보 없음)와 P5(이용료 정보 있음) 문항 세트만 지정하고, 반복되는 로짓 적합·WTP 계산·CSV 저장 로직은 이 파일에서 관리

from pathlib import Path
from numpy.linalg import LinAlgError
from statsmodels.tools.sm_exceptions import PerfectSeparationError, ConvergenceWarning, HessianInversionWarning
from scipy.stats import chi2, norm
from scipy import stats
from statsmodels.tools import add_constant
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from IPython.display import display
import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
import warnings


def run_first_stage_logit(df, result_path, q1_col, q2_yes_col, q2_no_col, label="전체"):
    """Bid-only/Bid+공변량 로짓표를 단일경계·이중경계 기준으로 저장"""
    # 전체(회원+접속자 합산) WTP 선형로짓 분석 - 안정화(중간 강도)
    # 범주형 단순화 : 기본 유지 (희소범주만 통합)
    # Bid : 10만 단위 스케일 (기존 유지)
    # 완전분리/불균형/수치불안 : 중간 강도 처리
    
    # 경고 무시
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # 1) 컬럼 매핑
    COLS = {
        "Gen": "P1-Q1",
        "Age": "P1-Q2",
        "Edu": "P1-Q5",
        "Pos": "P1-Q4",
        "Res": "P1-Q6",
        "Rti": "P2-Q1",
        "Inf": "P2-Q2"  # Likert 1~5 (연속 취급)
    }
    
    FIRST_BID = "Info2"
    Y1_COL = q1_col
    Y2_YCOL = q2_yes_col
    Y2_NCOL = q2_no_col
    
    # 2) 유틸
    def to_numeric_safe(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
        s = re.sub(r"[^0-9\.\-]", "", str(x)).replace(",", "")
        if s in ("", "-", ".", "-."): return np.nan
        try: return float(s)
        except Exception: return np.nan

    def compress_rare_categories(df_in: pd.DataFrame, cols, min_count=5, other_label="기타"):
        """희소 범주를 '기타'로 병합 (중간 강도)"""
        df = df_in.copy()
        for c in cols:
            if c not in df.columns: 
                continue
            ser = df[c].astype("string")
            vc = ser.value_counts(dropna=False)
            rare = set(vc[vc < min_count].index)
            df[c] = ser.map(lambda v: ("미응답" if pd.isna(v) else (other_label if v in rare else v)))
        return df

    def encode_categoricals(work: pd.DataFrame, cat_vars, y_series: pd.Series, min_count=5, imb_thresh=0.98):
        """더미화 + (중간 강도) 완전/준완전 분리·심한 불균형 제거"""
        out_block = {}
        if not cat_vars:
            return pd.DataFrame(index=work.index), out_block

        w = compress_rare_categories(work, cat_vars, min_count=min_count, other_label="기타")
        for c in cat_vars:
            if c in w.columns:
                w[c] = w[c].astype("string").fillna("미응답")

        dummies = pd.get_dummies(w[cat_vars].astype("string"), drop_first=True, dummy_na=False).astype(float)
        for v in cat_vars:
            out_block[v] = [col for col in dummies.columns if col.startswith(v + "_")]
        # (1) 완전/준완전 분리 제거
        keep = []
        yv = y_series.loc[dummies.index]
        for c in list(dummies.columns):
            x = dummies[c]
            if x.nunique(dropna=True) <= 1:
                continue
            y_on = yv[x == 1]
            y_off= yv[x == 0]
            if y_on.size > 0 and y_off.size > 0:
                if y_on.nunique(dropna=True) < 2 or y_off.nunique(dropna=True) < 2:
                    # 준완전 분리
                    continue
            # (2) 심한 불균형(1/0 거의 한쪽) 제거
            p1 = (x == 1).mean()
            if p1 <= (1 - imb_thresh) or p1 >= imb_thresh:
                continue
            keep.append(c)
        dummies = dummies[keep]
        # block 갱신
        for k in list(out_block.keys()):
            out_block[k] = [c for c in out_block[k] if c in dummies.columns]
            if not out_block[k]:
                del out_block[k]
        # (3) 고상관(>=0.999) 더미 제거
        if dummies.shape[1] > 1:
            corr = dummies.corr().abs()
            to_drop = set()
            cols = list(dummies.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if corr.iloc[i, j] >= 0.999:
                        to_drop.add(cols[j])
            if to_drop:
                dummies = dummies.drop(columns=list(to_drop), errors="ignore")
                for k in list(out_block.keys()):
                    out_block[k] = [c for c in out_block[k] if c in dummies.columns]
                    if not out_block[k]:
                        del out_block[k]

        return dummies, out_block

    def drop_constant_cols(X: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
        return X[keep]

    def drop_perfect_separation_cols(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """이진 더미가 y를 완전 예측하면 제거 (중간 강도)"""
        X2 = X.copy()
        for c in list(X2.columns):
            col = X2[c]
            if set(pd.unique(col.dropna())).issubset({0,1}):
                d1 = y[col == 1]; d0 = y[col == 0]
                if d1.size > 0 and d0.size > 0:
                    m1 = d1.mean(); m0 = d0.mean()
                    if (m1 == 1 and m0 == 0) or (m1 == 0 and m0 == 1):
                        X2 = X2.drop(columns=[c])
        return X2
    
    # 3) 전처리
    if FIRST_BID in df.columns:
        df[FIRST_BID] = df[FIRST_BID].apply(to_numeric_safe)
    for c in [Y1_COL, Y2_YCOL, Y2_NCOL]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    
    # 4) 분석용 데이터셋 구성
    def build_single_bounded(df_src: pd.DataFrame) -> pd.DataFrame:
        sub = df_src[[FIRST_BID, Y1_COL] + [v for v in COLS.values() if v in df_src.columns]].copy()
        sub.rename(columns={COLS[k]: k for k in COLS if COLS[k] in sub.columns}, inplace=True)
        sub["Bid"] = sub[FIRST_BID]  # Bid 원본 유지 (스케일링은 설계행렬에서)
        sub["Y"] = np.where(sub[Y1_COL] == "예", 1, np.where(sub[Y1_COL] == "아니오", 0, np.nan))
        return sub.dropna(subset=["Bid", "Y"])

    def build_double_bounded(df_src: pd.DataFrame) -> pd.DataFrame:
        need = [FIRST_BID, Y1_COL, Y2_YCOL, Y2_NCOL] + [v for v in COLS.values() if v in df_src.columns]
        sub = df_src[need].copy()
        sub.rename(columns={COLS[k]: k for k in COLS if COLS[k] in sub.columns}, inplace=True)
        sub["Bid"] = np.where(sub[Y1_COL] == "예", sub[FIRST_BID]*2, np.where(sub[Y1_COL] == "아니오", sub[FIRST_BID]*0.5, np.nan))
        sub["Y"] = np.nan
        mY = sub[Y1_COL] == "예"
        sub.loc[mY, "Y"] = np.where(sub.loc[mY, Y2_YCOL]=="예", 1, np.where(sub.loc[mY, Y2_YCOL]=="아니오", 0, np.nan))
        mN = sub[Y1_COL] == "아니오"
        sub.loc[mN, "Y"] = np.where(sub.loc[mN, Y2_NCOL]=="예", 1, np.where(sub.loc[mN, Y2_NCOL]=="아니오", 0, np.nan))
        return sub.dropna(subset=["Bid", "Y"])
    
    # 5) 설계행렬 (Design Matrix)
    def make_design(df_model: pd.DataFrame, with_covariates: bool, rare_min_count=5):
        work = df_model.copy()
        y = pd.to_numeric(work["Y"], errors="coerce")
        # Bid: 10만 단위 스케일(기존 유지)
        X = pd.DataFrame({"Bid": pd.to_numeric(work["Bid"], errors="coerce")/100_000.0}, index=work.index)
        block_cols = {}

        if with_covariates:
            cat_vars = [c for c in ["Gen", "Age", "Edu", "Pos", "Res", "Rti"] if c in work.columns]
            if cat_vars:
                dummies, block_map = encode_categoricals(work, cat_vars, y, min_count=rare_min_count, imb_thresh=0.98)
                if dummies.shape[1] > 0:
                    X = pd.concat([X, dummies], axis=1)
                    block_cols.update(block_map)

            if "Inf" in work.columns:
                inf = pd.to_numeric(work["Inf"].apply(to_numeric_safe), errors="coerce")
                if inf.notna().sum() > 0:
                    std = inf.std(ddof=0)
                    X["Inf"] = (inf - inf.mean()) / (std if (std and np.isfinite(std)) else 1.0)
                    block_cols["Inf"] = ["Inf"]

        valid = y.notna()
        for c in X.columns:
            valid &= X[c].notna()
        X, y = X.loc[valid], y.loc[valid]

        X = drop_constant_cols(X)
        X = drop_perfect_separation_cols(X, y)

        if with_covariates:
            for k in list(block_cols.keys()):
                block_cols[k] = [c for c in block_cols[k] if c in X.columns]
                if not block_cols[k]: del block_cols[k]

        return y, X, block_cols
    
    # 6) 로그우도/모형 통계
    def _ll_null(y):
        X0 = sm.add_constant(pd.DataFrame(index=y.index), has_constant="add")
        res0 = sm.Logit(y, X0).fit(disp=False, maxiter=400, method="lbfgs")
        return float(res0.llf)

    def _loglike_binom(y, linpred):
        lp = np.clip(linpred, -35, 35)
        p  = 1/(1+np.exp(-lp))
        eps = 1e-12
        p  = np.clip(p, eps, 1-eps)
        return float(np.sum(y*np.log(p) + (1-y)*np.log(1-p)))

    def _cov_from_hessian(model, params, ridge_eps=1e-4):
        H = model.hessian(params)
        k = H.shape[0]
        try:
            cov = np.linalg.pinv(-H + ridge_eps*np.eye(k))
        except Exception:
            cov = np.full((k,k), np.nan)
        return cov
    
    # 7) Logit 적합 (경고 최소화/안정 풀백)
    def fit_logit(y, X):
        if len(y) == 0 or y.nunique() < 2:
            return None

        Xc = sm.add_constant(X, has_constant="add")
        # (A) GLM 정규화 (약한 L2)
        try:
            glm_reg = sm.GLM(y, Xc, family=sm.families.Binomial())
            res_gr = glm_reg.fit_regularized(alpha=1e-3, L1_wt=0.0, maxiter=2000)
            params = pd.Series(res_gr.params, index=Xc.columns, dtype="float64")
            cov = _cov_from_hessian(sm.Logit(y, Xc), params, ridge_eps=1e-4)
            bse = pd.Series(np.sqrt(np.clip(np.diag(cov), 0, None)), index=Xc.columns, dtype="float64")
            z = params / bse.replace(0, np.nan)
            pvals = pd.Series(2*(1 - norm.cdf(np.abs(z))), index=Xc.columns, dtype="float64")
            llf = _loglike_binom(y, np.dot(Xc, params))
            try:
                lr = 2*(llf - _ll_null(y))
            except Exception:
                lr = np.nan
            aic = -2*llf + 2*len(params)
            return {"params": params, "bse": bse, "pvalues": pvals,
                    "llf": llf, "lr": lr, "aic": aic, "Xc_cols": list(Xc.columns)}
        except Exception:
            pass
        # (B) 표준 Logit (MLE)
        for meth in ["lbfgs", "newton", "bfgs", "ncg"]:
            try:
                res = sm.Logit(y, Xc).fit(disp=False, maxiter=800, method=meth)
                res_r = res.get_robustcov_results(cov_type="HC1")
                llf = float(res.llf); aic = float(res.aic)
                try:
                    lr = 2*(llf - _ll_null(y))
                except Exception:
                    lr = np.nan
                return {"params": res_r.params, "bse": res_r.bse, "pvalues": res_r.pvalues,
                        "llf": llf, "lr": lr, "aic": aic, "Xc_cols": list(Xc.columns)}
            except Exception:
                pass
        # (C) GLM Binomial (Logit)
        try:
            glm = sm.GLM(y, Xc, family=sm.families.Binomial())
            res_g = glm.fit(maxiter=1000)
            res_r = res_g.get_robustcov_results(cov_type="HC1")
            params, bse, pvals = res_r.params, res_r.bse, res_r.pvalues
            llf = _loglike_binom(y, np.dot(Xc, params))
            try:
                lr = 2*(llf - _ll_null(y))
            except Exception:
                lr = np.nan
            aic = -2*llf + 2*len(params)
            return {"params": params, "bse": bse, "pvalues": pvals,
                    "llf": llf, "lr": lr, "aic": aic, "Xc_cols": list(Xc.columns)}
        except Exception:
            pass
        # (D) 정규화 Logit (L2)
        try:
            model = sm.Logit(y, Xc)
            res_reg = model.fit_regularized(L1_wt=0.0, alpha=1e-3, maxiter=6000, trim_mode="off", cnvrg_tol=1e-10, disp=False)
            params = pd.Series(res_reg.params, index=Xc.columns, dtype="float64")
            cov = _cov_from_hessian(model, params, ridge_eps=1e-4)
            se  = pd.Series(np.sqrt(np.clip(np.diag(cov), 0, None)), index=Xc.columns, dtype="float64")
            z   = params / se.replace(0, np.nan)
            pvals = pd.Series(2*(1 - norm.cdf(np.abs(z))), index=Xc.columns, dtype="float64")
            llf = _loglike_binom(y, np.dot(Xc, params))
            try:
                lr = 2*(llf - _ll_null(y))
            except Exception:
                lr = np.nan
            aic = -2*llf + 2*len(params)
            return {"params": params, "bse": se, "pvalues": pvals,
                    "llf": llf, "lr": lr, "aic": aic, "Xc_cols": list(Xc.columns)}
        except Exception:
            return None
    
    # 8) 블록 LR p-value (범주형 묶음 단위)
    def block_lr_pvalue(y, X, block_cols):
        out = {}
        full = fit_logit(y, X)
        if full is None:
            return out, None
        llf_full = full["llf"]
        for var, cols in block_cols.items():
            if not cols:
                out[var] = np.nan
                continue
            X_red = X.drop(columns=cols, errors="ignore")
            red = fit_logit(y, X_red)
            if red is None or not np.isfinite(llf_full) or not np.isfinite(red["llf"]):
                out[var] = np.nan
                continue
            df_chi = (len(full["Xc_cols"]) - 1) - (len(red["Xc_cols"]) - 1)
            LR = 2*(llf_full - red["llf"])
            out[var] = 1 - chi2.cdf(LR, df_chi) if df_chi > 0 else np.nan
        return out, full
    
    # 9) 모형 요약표 생성
    def summarize_model_block(y, X, block_cols, row_order):
        p_block, full = block_lr_pvalue(y, X, block_cols)
        out = pd.DataFrame(index=row_order, columns=["Coefficient", "StdErr", "p-value"], dtype=float)

        if full is None:
            return out

        res = full
        params = res["params"]; bse = res["bse"]; pvals = res["pvalues"]

        out.loc["상수"] = [params.get("const", np.nan), bse.get("const", np.nan), pvals.get("const", np.nan)]
        out.loc["Bid"] = [params.get("Bid", np.nan), bse.get("Bid", np.nan), pvals.get("Bid", np.nan)]

        for var in ["Gen", "Age", "Edu", "Pos", "Res", "Rti"]:
            cols = block_cols.get(var, [])
            if cols:
                out.loc[var, "Coefficient"] = pd.to_numeric(params.reindex(cols), errors="coerce").mean()
                out.loc[var, "StdErr"] = pd.to_numeric(bse.reindex(cols), errors="coerce").mean()
                out.loc[var, "p-value"] = p_block.get(var, np.nan)
            else:
                out.loc[var] = [np.nan, np.nan, np.nan]

        if "Inf" in block_cols and block_cols["Inf"]:
            out.loc["Inf"] = [params.get("Inf", np.nan), bse.get("Inf", np.nan), pvals.get("Inf", np.nan)]
        else:
            out.loc["Inf"] = [np.nan, np.nan, np.nan]

        out.loc["Log likelihood"] = [res["llf"], np.nan, np.nan]
        out.loc["χ²"] = [res["lr"], np.nan, np.nan]
        out.loc["Akaike I.C."] = [res["aic"], np.nan, np.nan]
        return out

    def make_final_table(sb_b, sb_c, db_b, db_c):
        cols = pd.MultiIndex.from_product(
            [["단일경계", "이중경계"],
             ["Bid 금액만 포함", "Bid 금액과 공변량 포함"],
             ["Coefficient", "Standard error", "p-value"]],
            names=["구분", "세부", "통계"]
        )
        base_index = ["상수", "Bid", "Gen", "Age", "Edu", "Pos", "Res", "Rti", "Inf", "Log likelihood", "χ²", "Akaike I.C."]

        final = pd.DataFrame(index=base_index, columns=cols, dtype="float64")

        def _num(a):
            return pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype="float64")

        def put(block, gubun, detail):
            final.loc[:, (gubun, detail, "Coefficient")] = _num(block["Coefficient"].values)
            final.loc[:, (gubun, detail, "Standard error")] = _num(block["StdErr"].values)
            final.loc[:, (gubun, detail, "p-value")] = _num(block["p-value"].values)

        put(sb_b, "단일경계", "Bid 금액만 포함")
        put(sb_c, "단일경계", "Bid 금액과 공변량 포함")
        put(db_b, "이중경계", "Bid 금액만 포함")
        put(db_c, "이중경계", "Bid 금액과 공변량 포함")

        return final
    
    # 10) 실행 함수 구성
    def run_all(df_all: pd.DataFrame, outdir: Path, label="전체"):
        sb = build_single_bounded(df_all)
        db = build_double_bounded(df_all)

        rows = ["상수", "Bid", "Gen", "Age", "Edu", "Pos", "Res", "Rti", "Inf", "Log likelihood", "χ²", "Akaike I.C."]

        y_sb_b, X_sb_b, blk_sb_b = make_design(sb, with_covariates=False)
        y_sb_c, X_sb_c, blk_sb_c = make_design(sb, with_covariates=True,  rare_min_count=5)
        y_db_b, X_db_b, blk_db_b = make_design(db, with_covariates=False)
        y_db_c, X_db_c, blk_db_c = make_design(db, with_covariates=True,  rare_min_count=5)

        tab_sb_b = summarize_model_block(y_sb_b, X_sb_b, blk_sb_b, rows)
        tab_sb_c = summarize_model_block(y_sb_c, X_sb_c, blk_sb_c, rows)
        tab_db_b = summarize_model_block(y_db_b, X_db_b, blk_db_b, rows)
        tab_db_c = summarize_model_block(y_db_c, X_db_c, blk_db_c, rows)

        final = make_final_table(tab_sb_b, tab_sb_c, tab_db_b, tab_db_c).round(3)
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"WTP_Logit_Table_{label}.csv"
        final.to_csv(outpath, encoding="utf-8-sig")
        print(f"[{label}] 저장 완료:", outpath.resolve())
    
    # 11) 실행
    output_dir_all = Path(result_path) / "WTP_Logit_Outputs"
    output_dir_all.mkdir(parents=True, exist_ok=True)

    run_all(df, output_dir_all, label=label)


def run_second_stage_logit(df, result_path, q1_col, q2_yes_col, q2_no_col, alpha=0.10, label="전체"):
    """1차 full 모형의 블록 LR p-value 기준으로 유의 공변량만 남겨 재추정"""
    # 2차 분석 (A 방식): 블록 LR p≤0.10 변수만 선택
    # 전체(회원+접속자 통합)
    
    # 경고 무시
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # 1) 컬럼 매핑
    COLS = {
        "Gen": "P1-Q1",
        "Age": "P1-Q2",
        "Edu": "P1-Q5",
        "Pos": "P1-Q4",
        "Res": "P1-Q6",
        "Rti": "P2-Q1",
        "Inf": "P2-Q2"  # Likert 1~5 (연속 취급)
    }
    
    FIRST_BID = "Info2"
    Q1, Q2, Q5 = q1_col, q2_yes_col, q2_no_col
    
    # 2) 유틸
    def to_numeric_safe(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
        s = re.sub(r"[^0-9\.\-]", "", str(x)).replace(",", "")
        if s in ("", "-", ".", "-."): return np.nan
        try: return float(s)
        except Exception: return np.nan

    def yesno_to01(s):
        return s.map({"예":1, "아니오":0})

    def compress_rare_categories(df_in: pd.DataFrame, cols, min_count=5, other_label="기타"):
        df = df_in.copy()
        for c in cols:
            if c not in df.columns: continue
            ser = df[c].astype("string")
            vc  = ser.value_counts(dropna=False)
            rare = set(vc[vc < min_count].index)
            def _m(v):
                if pd.isna(v): return "미응답"
                return other_label if v in rare else v
            df[c] = ser.map(_m)
        return df

    def encode_categoricals(work: pd.DataFrame, cat_vars, y_series: pd.Series, min_count=5, imb_thresh=0.98):
        """더미 생성 + 희소/완전분리/극단불균형 제거 + 블록 매핑 반환"""
        out_block = {}
        if not cat_vars:
            return pd.DataFrame(index=work.index), out_block

        w = compress_rare_categories(work, cat_vars, min_count=min_count, other_label="기타")
        for c in cat_vars:
            if c in w.columns:
                w[c] = w[c].astype("string").fillna("미응답")

        dummies = pd.get_dummies(w[cat_vars].astype("string"), drop_first=True, dummy_na=False).astype(float)
        for v in cat_vars:
            out_block[v] = [col for col in dummies.columns if col.startswith(v + "_")]
        # 완전분리/극단불균형 제거
        keep = []
        yv = y_series.loc[dummies.index]
        for c in list(dummies.columns):
            x = dummies[c]
            if x.nunique(dropna=True) <= 1:
                continue
            y1 = yv[x==1]
            y0 = yv[x==0]
            if y1.size>0 and y0.size>0 and (y1.nunique()<2 or y0.nunique()<2):
                continue
            p1 = (x==1).mean()
            if p1 < (1-imb_thresh) or p1 > imb_thresh:
                continue
            keep.append(c)
        dummies = dummies[keep]
        # 고상관(>=0.999) 제거
        if dummies.shape[1] > 1:
            corr = dummies.corr().abs()
            drop = set()
            cols = list(dummies.columns)
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    if corr.iloc[i,j] >= 0.999:
                        drop.add(cols[j])
            if drop:
                dummies = dummies.drop(columns=list(drop), errors="ignore")
        # 블록 업데이트
        for k in list(out_block.keys()):
            out_block[k] = [c for c in out_block[k] if c in dummies.columns]
            if not out_block[k]:
                del out_block[k]

        return dummies, out_block

    def drop_constant_cols(X: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
        return X[keep]
    
    # 3) 데이터 구성
    def build_single_bounded(df_src: pd.DataFrame) -> pd.DataFrame:
        sub = df_src.copy()
        sub["Y"] = yesno_to01(sub[Q1])
        sub["Bid"] = pd.to_numeric(sub[FIRST_BID].apply(to_numeric_safe), errors="coerce") / 100_000.0
        return sub

    def implied_second_bid(row):
        b1 = to_numeric_safe(row[FIRST_BID])
        if pd.isna(b1): return np.nan
        r1 = row[Q1]
        if r1 == "예": return b1 * 2
        if r1 == "아니오": return b1 / 2
        return np.nan

    def build_double_bounded(df_src: pd.DataFrame) -> pd.DataFrame:
        sub = df_src.copy()
        sub["Bid2"] = sub.apply(implied_second_bid, axis=1)
        y2 = pd.Series(np.nan, index=sub.index)
        mY = sub[Q1] == "예"
        mN = sub[Q1] == "아니오"
        y2.loc[mY] = yesno_to01(sub.loc[mY, Q2])
        y2.loc[mN] = yesno_to01(sub.loc[mN, Q5])
        sub["Y"]   = y2
        sub["Bid"] = pd.to_numeric(sub["Bid2"].apply(to_numeric_safe), errors="coerce") / 100_000.0
        return sub
    
    # 4) 설계행렬 (블록 매핑 포함)
    def make_design(df_model: pd.DataFrame, with_covariates=True, rare_min_count=5):
        work = df_model.copy()
        # 문자열 정리
        for c in [Q1, Q2, Q5]:
            if c in work.columns:
                work[c] = work[c].astype(str).str.strip()

        y = pd.to_numeric(work["Y"], errors="coerce")
        X = pd.DataFrame({"Bid": pd.to_numeric(work["Bid"], errors="coerce")}, index=work.index)

        block_cols = {}
        if with_covariates:
            cat_vars = [c for c in ["Gen", "Age", "Edu", "Pos", "Res", "Rti"] if COLS.get(c) in work.columns]
            # 원자료 열명을 모델용으로 치환
            for short, raw in COLS.items():
                if raw in work.columns:
                    work[short] = work[raw]
            # 범주형 더미
            if cat_vars:
                dummies, block_map = encode_categoricals(work, cat_vars, y, min_count=rare_min_count, imb_thresh=0.98)
                if dummies.shape[1] > 0:
                    X = pd.concat([X, dummies], axis=1)
                    block_cols.update(block_map)
            # Inf (연속)
            if "Inf" in COLS and COLS["Inf"] in df.columns:
                inf = pd.to_numeric(work[COLS["Inf"]].apply(to_numeric_safe), errors="coerce")
                if inf.notna().sum() > 0:
                    std = inf.std(ddof=0)
                    X["Inf"] = (inf - inf.mean()) / (std if (std and np.isfinite(std)) else 1.0)
                    block_cols["Inf"] = ["Inf"]
        # 유효행 필터
        valid = y.notna()
        for c in X.columns: valid &= X[c].notna()
        X, y = X.loc[valid], y.loc[valid]
        X = drop_constant_cols(X)
        # 블록 열 재확정
        if with_covariates:
            for k in list(block_cols.keys()):
                block_cols[k] = [c for c in block_cols[k] if c in X.columns]
                if not block_cols[k]: del block_cols[k]

        return y, X, block_cols
    
    # 5) 적합
    def _loglike_binom(y, linpred):
        p = 1/(1+np.exp(-np.clip(linpred, -35, 35)))
        p = np.clip(p, 1e-12, 1-1e-12)
        return float(np.sum(y*np.log(p) + (1-y)*np.log(1-p)))

    def _ll_null(y):
        p = np.clip(np.nanmean(y), 1e-12, 1-1e-12)
        return float(np.sum(y*np.log(p) + (1-y)*np.log(1-p)))

    def _cov_from_hessian(model, params, ridge_eps=1e-4):
        H = model.hessian(params)
        k = H.shape[0]
        try:
            cov = np.linalg.pinv(-H + ridge_eps*np.eye(k))
        except Exception:
            cov = np.full((k,k), np.nan)
        return cov

    def fit_logit_stable(y, X):
        if len(y)==0 or y.nunique()<2:
            return None
        Xc = sm.add_constant(X, has_constant="add")
        # (A) GLM + 약한 L2
        try:
            glm = sm.GLM(y, Xc, family=sm.families.Binomial())
            res = glm.fit_regularized(alpha=1e-3, L1_wt=0.0, maxiter=3000)
            params = pd.Series(res.params, index=Xc.columns, dtype="float64")
            cov = _cov_from_hessian(sm.Logit(y, Xc), params, ridge_eps=1e-4)
            bse = pd.Series(np.sqrt(np.clip(np.diag(cov), 0, None)), index=Xc.columns, dtype="float64")
            z = params / bse.replace(0, np.nan)
            pvals = pd.Series(2*(1 - norm.cdf(np.abs(z))), index=Xc.columns, dtype="float64")
            llf = _loglike_binom(y, np.dot(Xc, params))
            try: lr = 2*(llf - _ll_null(y))
            except Exception: lr = np.nan
            aic = -2*llf + 2*len(params)
            return {"params": params, "bse": bse, "pvalues": pvals, "llf": llf, "lr": lr, "aic": aic, "cols": list(Xc.columns)}
        except Exception:
            pass
        # (B) 표준 Logit (MLE)
        for meth in ["lbfgs", "newton", "bfgs", "ncg"]:
            try:
                res = sm.Logit(y, Xc).fit(disp=False, maxiter=1200, method=meth)
                rob = res.get_robustcov_results(cov_type="HC1")
                return {"params": rob.params, "bse": rob.bse, "pvalues": rob.pvalues,
                        "llf": float(res.llf), "lr": float(res.llr), "aic": float(res.aic),
                        "cols": list(Xc.columns)}
            except Exception:
                pass
        # (C) 정규화 Logit (L2)
        try:
            model = sm.Logit(y, Xc)
            res = model.fit_regularized(L1_wt=0.0, alpha=1e-3, maxiter=6000, trim_mode="off", cnvrg_tol=1e-10, disp=False)
            params = pd.Series(res.params, index=Xc.columns, dtype="float64")
            cov = _cov_from_hessian(model, params, ridge_eps=1e-4)
            se  = pd.Series(np.sqrt(np.clip(np.diag(cov), 0, None)), index=Xc.columns, dtype="float64")
            z   = params / se.replace(0, np.nan)
            pvals = pd.Series(2*(1 - norm.cdf(np.abs(z))), index=Xc.columns, dtype="float64")
            llf = _loglike_binom(y, np.dot(Xc, params))
            try: lr = 2*(llf - _ll_null(y))
            except Exception: lr = np.nan
            aic = -2*llf + 2*len(params)
            return {"params": params, "bse": se, "pvalues": pvals, "llf": llf, "lr": lr, "aic": aic, "cols": list(Xc.columns)}
        except Exception:
            return None
    
    # 6) 블록 LR p-value 계산 (변수 단위 선택)
    def block_lr_pvalues(y, X, block_cols):
        out = {}
        full = fit_logit_stable(y, X)
        if (full is None) or (not np.isfinite(full.get("llf", np.nan))):
            return out, full

        llf_full = full["llf"]
        k_full   = len(full["cols"])

        for var, cols in block_cols.items():
            if not cols: 
                out[var] = np.nan
                continue
            X_red = X.drop(columns=cols, errors="ignore")
            red = fit_logit_stable(y, X_red)
            if (red is None) or (not np.isfinite(red.get("llf", np.nan))):
                out[var] = np.nan
                continue
            # 자유도 = 제거된 계수 수
            k_red = len(red["cols"])
            df_chi = (k_full - k_red)
            LR = 2*(llf_full - red["llf"])
            out[var] = 1 - chi2.cdf(LR, df_chi) if df_chi>0 else np.nan

        return out, full
    
    # 7) 표 생성
    ROWS = ["상수",  "Bid", "Gen",  "Age", "Edu",  "Pos", "Res",  "Rti", "Inf",  "Log likelihood", "χ²", "Akaike I.C."]

    def summarize_selected(y, X, block_cols, alpha=0.10):
        """full 적합→ 블록 LR p≤alpha 선택 → 선택모형 재적합 → 표 블록 반환"""
        p_block, full = block_lr_pvalues(y, X, block_cols)
        # 선택 변수(블록)
        selected_blocks = [var for var, p in p_block.items() if (p is not None) and (not pd.isna(p)) and (p <= alpha)]
        # keep columns
        keep = ["Bid"]
        for var in selected_blocks:
            keep += block_cols.get(var, [])
        keep = [c for c in keep if c in X.columns]

        res_final = fit_logit_stable(y, X[keep]) if keep else fit_logit_stable(y, X[["Bid"]])
        # 표 구성
        out = pd.DataFrame(index=ROWS, columns=["Coefficient", "StdErr", "p-value"], dtype="float64")
        if res_final is None:
            return out, p_block, selected_blocks

        params = res_final["params"]; bse = res_final["bse"]; pvals = res_final["pvalues"]
        # 상수/ Bid
        out.loc["상수"] = [params.get("const", np.nan), bse.get("const", np.nan), pvals.get("const", np.nan)]
        out.loc["Bid"]  = [params.get("Bid",   np.nan), bse.get("Bid",   np.nan), pvals.get("Bid",   np.nan)]
        # 블록별 계수/SE 평균, p-value는 블록 LR p
        for var in ["Gen", "Age", "Edu",  "Pos", "Res",  "Rti", "Inf"]:
            cols = [c for c in block_cols.get(var, []) if c in params.index]
            if cols:
                out.loc[var, "Coefficient"] = float(pd.to_numeric(params.reindex(cols), errors="coerce").mean())
                out.loc[var, "StdErr"] = float(pd.to_numeric(bse.reindex(cols),    errors="coerce").mean())
                out.loc[var, "p-value"] = p_block.get(var, np.nan)
            else:
                out.loc[var] = [np.nan, np.nan, np.nan]

        out.loc["Log likelihood","Coefficient"] = res_final.get("llf", np.nan)
        out.loc["χ²","Coefficient"] = res_final.get("lr",  np.nan)
        out.loc["Akaike I.C.","Coefficient"] = res_final.get("aic", np.nan)

        return out.round(3), p_block, selected_blocks

    def combine_table(sb_block, db_block, alpha=0.10):
        tag = f"Bid 금액과 공변량 포함(선택: block LR p≤{alpha:.2f})"
        cols = pd.MultiIndex.from_product(
            [["단일경계","이중경계"], [tag], ["Coefficient","Standard error","p-value"]],
            names=["구분","세부","통계"]
        )
        final = pd.DataFrame(index=ROWS, columns=cols, dtype="float64")
        def put(block, gubun):
            final.loc[:, (gubun, tag, "Coefficient")] = block["Coefficient"]
            final.loc[:, (gubun, tag, "Standard error")] = block["StdErr"]
            final.loc[:, (gubun, tag, "p-value")] = block["p-value"]
        put(sb_block, "단일경계")
        put(db_block, "이중경계")
        return final
    
    # 8) 실행 함수 구성
    def run_second_stage_blockLR_all(df_all: pd.DataFrame, outdir: Path, alpha=0.10, label="전체"):
        # 숫자/텍스트 정리
        if FIRST_BID in df_all.columns:
            df_all[FIRST_BID] = df_all[FIRST_BID].apply(to_numeric_safe)
        for c in [Q1, Q2, Q5]:
            if c in df_all.columns:
                df_all[c] = df_all[c].astype(str).str.strip()
        # 데이터 구성
        sb = build_single_bounded(df_all)
        db = build_double_bounded(df_all)
        # 설계행렬
        y_sb, X_sb, blk_sb = make_design(sb, with_covariates=True, rare_min_count=5)
        y_db, X_db, blk_db = make_design(db, with_covariates=True, rare_min_count=5)
        # 선택모형 요약표
        tab_sb, p_sb, sel_sb = summarize_selected(y_sb, X_sb, blk_sb, alpha=alpha)
        tab_db, p_db, sel_db = summarize_selected(y_db, X_db, blk_db, alpha=alpha)

        final = combine_table(tab_sb, tab_db, alpha=alpha)

        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"WTP_Logit_Table_2nd_{label}.csv"
        final.to_csv(outpath, encoding="utf-8-sig")
        print(f"[2차(블록LR)-{label}] 저장 완료:", outpath.resolve())
        # 선택된 변수 로그
        print("단일경계 선택 블록:", sel_sb)
        print("이중경계 선택 블록:", sel_db)
    
    # 9) 실행
    output_dir_2nd = Path(result_path) / "WTP_Logit_Outputs_2nd"
    output_dir_2nd.mkdir(parents=True, exist_ok=True)
    
    # 실행 (alpha=0.10 기준 선택)
    run_second_stage_blockLR_all(df, output_dir_2nd, alpha=alpha, label=label)


def estimate_wtp_selected_covariates(
    df,
    result_path,
    q1_col,
    q2_yes_col,
    q2_no_col,
    alpha=0.05,
    caption="② 유의 공변량만 포함한 추정치를 사용한 WTP 추정 (전체)",
):
    """유의 공변량 선택 모형으로 표본 WTP 평균·중앙값·절단평균 계산"""
    
    # 경고 무시
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # 1) 입력 컬럼 매핑
    FIRST_BID = "Info2"   # 첫 제시금액(원)
    Y1_COL = q1_col       # 첫 응답 (예/아니오)
    Y2_YCOL = q2_yes_col  # (Q1=예) 두번째 응답
    Y2_NCOL = q2_no_col   # (Q1=아니오) 두번째 응답
    
    # 공변량 (원자료 컬럼명 매핑)
    COV_MAP = {
        "Gen": "P1-Q1",  # 성별
        "Age": "P1-Q2",  # 나이
        "Edu": "P1-Q5",  # 학력
        "Pos": "P1-Q4",  # 직급
        "Res": "P1-Q6",  # 과제책임여부
        "Rti": "P2-Q1",  # 연구관련 시간 할애
        "Inf": "P2-Q2"   # 영향 정도
    }
    
    # 결과 저장 폴더
    outdir = Path(result_path) / "WTP_Logit_Outputs_3rd_Summary"
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 옵션
    ALPHA = alpha                # 유의수준 (2차: 유의 공변량만 선별)
    USE_OBSERVED_AMAX = True     # 표본 내 최대 제시금액을 A_max로 사용
    AMAX_FALLBACK = 1_200_000
    
    # 2) 전처리
    def to01_strong(s: pd.Series) -> pd.Series:
        """예/아니오·YES/NO·1/0 등 혼재 → 0/1 숫자"""
        t = s.astype(str).str.strip()
        m = {"예":1, "아니오":0, "YES":1, "NO":0, "Yes":1, "No":0, "yes":1, "no":0,
             "1":1, "0":0, "1.0":1, "0.0":0}
        out = t.map(m)
        # 남은 값은 숫자 캐스팅 시도
        out_num = pd.to_numeric(t, errors="coerce")
        out = out.fillna(out_num)
        return pd.to_numeric(out, errors="coerce")

    def add_const(X: pd.DataFrame) -> pd.DataFrame:
        return sm.add_constant(X, has_constant="add")

    def fit_logit_stable(y: pd.Series, X: pd.DataFrame):
        """
        표준 Logit → GLM Binomial → Regularized 순 폴백
        실패 시 None / 성공 시 statsmodels result-like 객체 반환
        """
        Xc = add_const(X)
        yv = pd.to_numeric(y, errors="coerce")
        valid = np.isfinite(Xc).all(1) & np.isfinite(yv)
        Xc, yv = Xc.loc[valid], yv.loc[valid]
        # 데이터 요건
        if yv.nunique() < 2 or Xc.shape[0] <= Xc.shape[1]:
            return None
        # 1) 표준 Logit
        try:
            return sm.Logit(yv, Xc).fit(disp=False, maxiter=1000)
        except Exception:
            pass
        # 2) GLM Binomial
        try:
            glm = sm.GLM(yv, Xc, family=sm.families.Binomial())
            return glm.fit(maxiter=1000)
        except Exception:
            pass
        # 3) Regularized (p-value는 NaN 처리)
        try:
            res_r = sm.Logit(yv, Xc).fit_regularized(L1_wt=0.0, alpha=1e-6, maxiter=6000, trim_mode="off", cnvrg_tol=1e-10, disp=False)
            class R:
                params = res_r.params
                nobs = len(yv)
                def cov_params(self):
                    bse = getattr(res_r, "bse", None)
                    if bse is None:
                        return pd.DataFrame(np.nan*np.ones((len(self.params),len(self.params))),
                                            index=self.params.index, columns=self.params.index)
                    return pd.DataFrame(np.diag(np.asarray(bse)**2),
                                        index=self.params.index, columns=self.params.index)
                @property
                def pvalues(self):
                    # 정규화 적합의 p-value는 신뢰하기 어려우므로 NaN 처리
                    return pd.Series([np.nan]*len(self.params), index=self.params.index)
                llf = np.nan; aic = np.nan
            return R()
        except Exception:
            return None

    def pick_significant(res, alpha=ALPHA):
        """상수/가격 제외, 유의 공변량만 선별 / p-value 없으면 빈 리스트"""
        if res is None: return []
        p = getattr(res, "pvalues", None)
        if p is None or (isinstance(p, pd.Series) and p.isna().all()):
            return []
        return [c for c in p.index if c not in ("const", "Bid") and (p[c] < alpha)]

    def alpha_beta_from(res, X_use: pd.DataFrame, used_covs: list):
        """
        공분산 행렬에서 delta-method용 2x2 블록 V(alpha*, beta) 구성.
        """
        if res is None: return np.nan, np.nan, None
        params = res.params
        cov = res.cov_params()
        if isinstance(cov, np.ndarray):
            cov = pd.DataFrame(cov, index=params.index, columns=params.index)
        if "Bid" not in params.index:
            return np.nan, np.nan, None

        means = X_use[used_covs].mean() if used_covs else pd.Series(dtype=float)
        a_vec = pd.Series(0.0, index=params.index)
        if "const" in a_vec.index: a_vec["const"] = 1.0
        for z in used_covs:
            if z in a_vec.index: a_vec[z] = float(means.get(z, 0.0))

        alpha = float(np.dot(a_vec, params))   # alpha*
        beta = -float(params["Bid"])           # beta = -gamma_Bid

        if cov is None: 
            return alpha, beta, None
        var_alpha = float(a_vec @ cov.loc[a_vec.index, a_vec.index] @ a_vec)
        cov_ab = float(a_vec @ cov.loc[a_vec.index, "Bid"])
        var_beta  = float(cov.loc["Bid","Bid"])
        V = np.array([[var_alpha, -cov_ab], [-cov_ab, var_beta]])
        return alpha, beta, V

    def se_delta(grad, V):
        if V is None or not np.isfinite(V).all():
            return np.nan
        return float(np.sqrt(np.maximum(0.0, grad @ V @ grad)))

    def wtp_mean(alpha, beta, V):
        if not np.isfinite(alpha) or not np.isfinite(beta) or beta == 0:
            return np.nan, np.nan
        m = np.log1p(np.exp(alpha)) / beta
        g = np.array([ 1/(1+np.exp(-alpha))/beta,  -np.log1p(np.exp(alpha))/(beta**2) ])
        return m, se_delta(g, V)

    def wtp_median(alpha, beta, V):
        if not np.isfinite(alpha) or not np.isfinite(beta) or beta == 0:
            return np.nan, np.nan
        md = alpha / beta
        g = np.array([ 1/beta,  -alpha/(beta**2) ])
        return md, se_delta(g, V)

    def wtp_trunc_mean(alpha, beta, Amax, V):
        if not np.isfinite(alpha) or not np.isfinite(beta) or beta == 0:
            return np.nan, np.nan
        t1 = np.log1p(np.exp(alpha))
        t2 = np.log1p(np.exp(alpha - beta*Amax))
        tm = (t1 - t2) / beta
        s1 = 1/(1+np.exp(-alpha))
        s2 = 1/(1+np.exp(-(alpha - beta*Amax)))
        d_alpha = (s1 - s2)/beta
        d_beta = -tm/beta + (Amax*s2)/beta
        g = np.array([d_alpha, d_beta])
        return tm, se_delta(g, V)

    def ci95(est, se):
        if not np.isfinite(est) or not np.isfinite(se): 
            return (np.nan, np.nan)
        z = 1.96
        return (est - z*se, est + z*se)

    def r3(x): 
        return np.round(x, 3) if (x is not None) and np.isfinite(x) else np.nan

    def ci_str(lo, hi): 
        return f"({r3(lo)}, {r3(hi)})"
    
    # 3) 파생변수 생성 (매핑 기반)
    # Y_SB, Bid_SB (단일경계)
    # Y_DB, Bid_DB (이중경계; 2배/절반 규칙)
    # 공변량 숫자화
    # 단일경계
    df["Y_SB"]   = to01_strong(df[Y1_COL])
    df["Bid_SB"] = pd.to_numeric(df[FIRST_BID], errors="coerce")  # 스케일링 없음(원 단위)
    # 이중경계 (두번째 제시금액과 그 응답)
    def _second_bid_and_y(row):
        b1 = pd.to_numeric(row[FIRST_BID], errors="coerce")
        y1 = to01_strong(pd.Series([row[Y1_COL]])).iloc[0]
        if pd.isna(b1) or pd.isna(y1): 
            return pd.Series([np.nan, np.nan])
        if y1 == 1:
            b2 = 2.0 * b1
            y2 = to01_strong(pd.Series([row[Y2_YCOL]])).iloc[0]
        else:
            b2 = 0.5 * b1
            y2 = to01_strong(pd.Series([row[Y2_NCOL]])).iloc[0]
        return pd.Series([b2, y2])

    df[["Bid_DB","Y_DB"]] = df.apply(_second_bid_and_y, axis=1)
    # 공변량을 분석용 컬럼명으로 병합 (숫자화)
    for short, raw in COV_MAP.items():
        if raw in df.columns:
            df[short] = pd.to_numeric(df[raw], errors="coerce")
        else:
            df[short] = np.nan  # 빠진 컬럼은 NaN으로 채움

    COV_COLS = list(COV_MAP.keys())  # ["Gen", "Age", "Edu", "Pos", "Res", "Rti", "Inf"]
    
    # 4) 2차 적합 (유의 공변량만 포함) → alpha*, beta → WTP 계산
    # 사용된 공변량과 최종 파라미터 로깅
    def block_estimate_with_selection(y_col, bid_col, label_for_log=""):
        # 설계행렬 구성 (Bid + 공변량)
        y = to01_strong(df[y_col])
        X = pd.DataFrame({"Bid": pd.to_numeric(df[bid_col], errors="coerce")})
        for c in COV_COLS:
            X[c] = pd.to_numeric(df[c], errors="coerce")
        # 풀모형 적합
        res_full = fit_logit_stable(y, X)
        # 유의 공변량 선택 → Bid + 선택된 공변량만으로 재적합
        used_covs = pick_significant(res_full, alpha=ALPHA)
        X_use = X[["Bid"] + used_covs] if used_covs else X[["Bid"]]
        res = fit_logit_stable(y, X_use)
        # 실패 또는 Bid 누락 시, 최종 폴백: Bid만
        if (res is None) or ("Bid" not in getattr(res, "params", pd.Series([])).index):
            X_use = X[["Bid"]]
            res   = fit_logit_stable(y, X_use)
            used_covs = []
        # 디버그/점검용 로깅
        print(f"[{label_for_log}] 사용된 공변량 =", used_covs)
        try:
            print(f"[{label_for_log}] 최종 params =\n{res.params}\n")
        except Exception:
            print(f"[{label_for_log}] 최종 params = None\n")
        # alpha*, beta, V
        alpha, beta, V = alpha_beta_from(res, X_use, used_covs)
        # Amax
        bids = pd.to_numeric(df[bid_col], errors="coerce")
        Amax = float(np.nanmax(bids)) if USE_OBSERVED_AMAX else AMAX_FALLBACK
        # WTP (평균/중앙/절단평균) + SE + 95% CI
        m,  m_se  = wtp_mean(alpha, beta, V)
        md, md_se = wtp_median(alpha, beta, V)
        tm, tm_se = wtp_trunc_mean(alpha, beta, Amax, V)

        return {
            "WTP 평균": r3(m), "표준오차(평균)": r3(m_se), "95% 신뢰구간(평균)": ci_str(*ci95(m,  m_se)),
            "WTP 중앙값": r3(md),"표준오차(중앙값)": r3(md_se),"95% 신뢰구간(중앙)": ci_str(*ci95(md, md_se)),
            "WTP 절단된평균값": r3(tm), "표준오차(절단)": r3(tm_se), "95% 신뢰구간(절단)": ci_str(*ci95(tm, tm_se))
        }
    
    # 단일/이중 결과 (유의 공변량 포함 추정치 기반)
    sb = block_estimate_with_selection("Y_SB", "Bid_SB", label_for_log="단일경계")
    db = block_estimate_with_selection("Y_DB", "Bid_DB", label_for_log="이중경계")

    final = pd.DataFrame(
        [
            {"구분": "단일경계", **sb},
            {"구분": "이중경계", **db},
        ],
        columns=[
            "구분",
            "WTP 평균", "표준오차(평균)", "95% 신뢰구간(평균)",
            "WTP 중앙값", "표준오차(중앙값)", "95% 신뢰구간(중앙)",
            "WTP 절단된평균값", "표준오차(절단)", "95% 신뢰구간(절단)"
        ]
    )
    
    # 5) 저장/출력
    outpath = outdir / "1_WTP_표본추정_요약(유의공변량선택)_전체.csv"
    final.to_csv(outpath, index=False, encoding="utf-8-sig")
    display(final.style.set_caption(caption))
    print(f"저장 완료: {outpath}")


def estimate_wtp_bid_only(
    df,
    result_path,
    q1_col,
    q2_yes_col,
    q2_no_col,
    caption="②의 분석 중 bid 금액만 포함된 추정치를 사용한 WTP 추정 (전체)",
):
    """Bid만 포함한 모형으로 표본 WTP 평균·중앙값·절단평균 계산"""
    # Bid만 포함된 추정치를 사용한 WTP 추정 (전체)
    # 입력: df (원자료 DataFrame), result_path (상위 셀에서 지정)
    # 출력: 표(DataFrame) + CSV 저장
    
    # 1) 입력 컬럼 매핑
    FIRST_BID = "Info2"  # 첫 제시금액(원)
    Y1_COL = q1_col       # 첫 응답 (예/아니오)
    Y2_YCOL = q2_yes_col  # (Q1=예) 두번째 응답
    Y2_NCOL = q2_no_col   # (Q1=아니오) 두번째 응답
    
    # 2) 유틸
    def yesno_to01(s: pd.Series) -> pd.Series:
        t = s.astype(str).str.strip()
        m = {"예":1, "아니오":0, "YES":1, "NO":0, "Yes":1, "No":0,
             "yes":1, "no":0, "1":1, "0":0, "1.0":1, "0.0":0}
        out = t.map(m)
        # 숫자도 허용
        out = out.fillna(pd.to_numeric(t, errors="coerce"))
        return pd.to_numeric(out, errors="coerce")

    def add_const(X): 
        return sm.add_constant(X, has_constant="add")

    def fit_logit_bid_only(y: pd.Series, bid_series: pd.Series):
        """X=Bid(원)만 사용. 표준 Logit → GLM → Regularized 순 폴백"""
        X = pd.DataFrame({"Bid": pd.to_numeric(bid_series, errors="coerce")})
        Xc = add_const(X)
        yv = pd.to_numeric(y, errors="coerce")

        valid = np.isfinite(Xc).all(1) & np.isfinite(yv)
        Xc, yv = Xc.loc[valid], yv.loc[valid]

        if yv.nunique() < 2 or len(yv) <= Xc.shape[1]:
            return None, Xc
        # 1) 표준 Logit
        try:
            return sm.Logit(yv, Xc).fit(disp=False, maxiter=1000), Xc
        except Exception:
            pass
        # 2) GLM Binomial
        try:
            glm = sm.GLM(yv, Xc, family=sm.families.Binomial())
            return glm.fit(maxiter=1000), Xc
        except Exception:
            pass
        # 3) Regularized (SE는 근사 bse가 없을 수 있음 → NaN 가능)
        try:
            res_r = sm.Logit(yv, Xc).fit_regularized(L1_wt=0.0, alpha=1e-6, maxiter=6000, trim_mode="off", cnvrg_tol=1e-10, disp=False)
            class R:
                params = res_r.params
                nobs = len(yv)
                def cov_params(self):
                    bse = getattr(res_r, "bse", None)
                    if bse is None:
                        return pd.DataFrame(np.nan*np.ones((len(self.params),len(self.params))),
                                            index=self.params.index, columns=self.params.index)
                    return pd.DataFrame(np.diag(np.asarray(bse)**2),
                                        index=self.params.index, columns=self.params.index)
            return R(), Xc
        except Exception:
            return None, Xc

    def alpha_beta_from(res):
        if res is None or ("Bid" not in res.params.index):
            return np.nan, np.nan, None
        params = res.params
        cov = res.cov_params()
        if isinstance(cov, np.ndarray):
            cov = pd.DataFrame(cov, index=params.index, columns=params.index)

        alpha = float(params.get("const", 0.0))
        beta = -float(params["Bid"])

        if cov is None:
            return alpha, beta, None

        var_alpha = float(cov.loc["const", "const"]) if "const" in cov.index else 0.0
        cov_ab = float(cov.loc["const", "Bid"]) if ("const" in cov.index) else 0.0
        var_beta = float(cov.loc["Bid", "Bid"])
        V = np.array([[var_alpha, -cov_ab], [-cov_ab, var_beta]])
        return alpha, beta, V

    def se_delta(grad, V):
        if V is None or not np.isfinite(V).all():
            return np.nan
        return float(np.sqrt(np.maximum(0.0, grad @ V @ grad)))

    def wtp_mean(alpha, beta, V):
        if not np.isfinite(alpha) or not np.isfinite(beta) or beta == 0:
            return np.nan, np.nan
        m = np.log1p(np.exp(alpha)) / beta
        g = np.array([1/(1+np.exp(-alpha))/beta, -np.log1p(np.exp(alpha))/(beta**2)])
        return m, se_delta(g, V)

    def wtp_median(alpha, beta, V):
        if not np.isfinite(alpha) or not np.isfinite(beta) or beta == 0:
            return np.nan, np.nan
        md = alpha / beta
        g  = np.array([1/beta, -alpha/(beta**2)])
        return md, se_delta(g, V)

    def wtp_trunc_mean(alpha, beta, Amax, V):
        if not np.isfinite(alpha) or not np.isfinite(beta) or beta == 0:
            return np.nan, np.nan
        t1 = np.log1p(np.exp(alpha))
        t2 = np.log1p(np.exp(alpha - beta*Amax))
        tm = (t1 - t2) / beta
        s1 = 1/(1+np.exp(-alpha))
        s2 = 1/(1+np.exp(-(alpha - beta*Amax)))
        d_alpha = (s1 - s2)/beta
        d_beta = -tm/beta + (Amax*s2)/beta
        g = np.array([d_alpha, d_beta])
        return tm, se_delta(g, V)

    def ci95(est, se):
        if not np.isfinite(est) or not np.isfinite(se):
            return (np.nan, np.nan)
        z = 1.96
        return (est - z*se, est + z*se)

    def r3(x): 
        return np.round(x, 3) if (x is not None) and np.isfinite(x) else np.nan
    def ci_str(lo, hi): 
        return f"({r3(lo)}, {r3(hi)})"
    
    # 3) 단일/이중 경계 파생변수 (Bid-Only)
    # 단일경계: Y_SB, Bid_SB
    df["Y_SB"]  = yesno_to01(df[Y1_COL])
    df["Bid_SB"] = pd.to_numeric(df[FIRST_BID], errors="coerce")  # 스케일링 없음(원 단위)
    # 이중경계: Y_DB, Bid_DB (규칙: 예 → 2배, 아니오 → 1/2배)
    def make_db(row):
        b1 = pd.to_numeric(row[FIRST_BID], errors="coerce")
        y1 = yesno_to01(pd.Series([row[Y1_COL]])).iloc[0]
        if pd.isna(b1) or pd.isna(y1): 
            return pd.Series([np.nan, np.nan])
        if y1 == 1:
            b2 = 2.0*b1
            y2 = yesno_to01(pd.Series([row[Y2_YCOL]])).iloc[0]
        else:
            b2 = 0.5*b1
            y2 = yesno_to01(pd.Series([row[Y2_NCOL]])).iloc[0]
        return pd.Series([b2, y2])

    df[["Bid_DB","Y_DB"]] = df.apply(make_db, axis=1)
    
    # 4) 블록 계산 (단일/이중 각각)
    USE_OBSERVED_AMAX = True
    AMAX_FALLBACK = 1_200_000  # 원

    def block_bid_only(y_col, bid_col):
        # (1) 적합
        y = yesno_to01(df[y_col])
        bid = pd.to_numeric(df[bid_col], errors="coerce")
        res, Xc = fit_logit_bid_only(y, bid)
        # (2) alpha, beta, covariance
        alpha, beta, V = alpha_beta_from(res)
        # (3) A_max
        Amax = float(np.nanmax(bid)) if USE_OBSERVED_AMAX else AMAX_FALLBACK
        # (4) WTP (평균/중앙/절단평균) + SE + 95%CI
        m,  m_se  = wtp_mean(alpha, beta, V)
        md, md_se = wtp_median(alpha, beta, V)
        tm, tm_se = wtp_trunc_mean(alpha, beta, Amax, V)

        return {
            "WTP 평균": r3(m), "표준오차(평균)": r3(m_se), "95% 신뢰구간(평균)": ci_str(*ci95(m,  m_se)),
            "WTP 중앙값": r3(md), "표준오차(중앙값)": r3(md_se),"95% 신뢰구간(중앙)": ci_str(*ci95(md, md_se)),
            "WTP 절단된평균값": r3(tm), "표준오차(절단)": r3(tm_se), "95% 신뢰구간(절단)": ci_str(*ci95(tm, tm_se))
        }

    sb = block_bid_only("Y_SB", "Bid_SB")
    db = block_bid_only("Y_DB", "Bid_DB")

    final_bidonly = pd.DataFrame([
        {"구분": "단일경계", **sb},
        {"구분": "이중경계", **db},
    ], columns=[
        "구분",
        "WTP 평균", "표준오차(평균)", "95% 신뢰구간(평균)",
        "WTP 중앙값", "표준오차(중앙값)", "95% 신뢰구간(중앙)",
        "WTP 절단된평균값", "표준오차(절단)", "95% 신뢰구간(절단)"
    ])
    
    # 5) 저장
    outdir = Path(result_path) / "WTP_Logit_Outputs_3rd_Summary"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "2_WTP_표본추정_요약(BidOnly).csv"
    final_bidonly.to_csv(outpath, index=False, encoding="utf-8-sig")

    display(final_bidonly.style.set_caption(caption))
    print(f"저장 완료: {outpath}")
