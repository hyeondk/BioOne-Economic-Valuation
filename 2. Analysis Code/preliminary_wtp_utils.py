## 예비설문 WTP 분석 공통 함수 ##
# 노트북에는 분석 단계별 호출만 남기고, 로짓 적합/WTP 산출/RMSE 계산처럼 반복되거나 긴 로직은 이 파일에서 관리

from pathlib import Path
from numpy.linalg import LinAlgError
from scipy.stats import chi2, norm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning, PerfectSeparationWarning
from IPython.display import display
import contextlib
import io
import re
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=HessianInversionWarning)
warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@contextlib.contextmanager
def quiet_statsmodels():
    """로짓 적합 과정에서 발생하는 statsmodels 경고와 최적화 로그를 노트북 출력에서 숨기기"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yield


@contextlib.contextmanager
def suppress_model_warnings():
    """분석 셀 전체에서 statsmodels/numpy 경고가 노트북 출력으로 새지 않게 막기"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("ignore", message=".*Inverting hessian failed.*")
        warnings.filterwarnings("ignore", message=".*Perfect separation.*")
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=HessianInversionWarning)
        warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            # 저장 완료 메시지는 stdout으로 유지하고, 경고가 쓰이는 stderr만 숨김
            with contextlib.redirect_stderr(io.StringIO()):
                yield


# 1차 WTP 로짓모형: 단일/이중경계 각각 Bid-only와 Bid+공변량 사양 추정
INFO_COL = "Info1"
BID_COL = "Info2"
Q1_COL, Q2_YES_COL, Q2_NO_COL = "P4-Q1", "P4-Q2", "P4-Q5"

COV_MAP = {
    "Gen": "P1-Q1",  # 성별
    "Age": "P1-Q2",  # 나이
    "Edu": "P1-Q5",  # 학력
    "Pos": "P1-Q4",  # 직급
    "Res": "P1-Q6",  # 과제책임여부
    "Rti": "P2-Q1",  # 연구관련 시간 할애
    "Inf": "P2-Q2",  # BioOne 영향 정도(1~5)
}
MODEL_ROWS = ["상수", "Bid", "Gen", "Age", "Edu", "Pos", "Res", "Rti", "Inf", "Log likelihood", "χ²", "Akaike I.C."]


def to_numeric_safe(value):
    """숫자와 쉼표/원 단위가 섞인 값을 float로 변환"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = re.sub(r"[^0-9\.\-]", "", str(value).replace(",", ""))
    return np.nan if text in {"", "-", ".", "-."} else float(text)


def yn_to01(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"예", "Yes", "1", "1.0"}:
        return 1.0
    if text in {"아니오", "No", "0", "0.0"}:
        return 0.0
    return pd.to_numeric(text, errors="coerce")


def collapse_rare_categories(frame: pd.DataFrame, cols, min_count=3, other_label="기타") -> pd.DataFrame:
    """표본 수가 작은 범주를 묶어 더미 폭주와 완전분리 위험 줄이기"""
    out = frame.copy()
    for col in cols:
        if col not in out.columns:
            continue
        ser = out[col].astype("string")
        rare = set(ser.value_counts(dropna=False).loc[lambda s: s < min_count].index)
        out[col] = ser.map(lambda v: np.nan if pd.isna(v) else other_label if v in rare else v)
    return out


def add_wtp_response_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """단일경계/이중경계 로짓에 필요한 Y와 Bid를 원자료에 추가"""
    out = frame.copy()
    out[BID_COL] = out[BID_COL].map(to_numeric_safe)
    for col in [Q1_COL, Q2_YES_COL, Q2_NO_COL]:
        out[col] = out[col].astype(str).str.strip()

    out["Y_SB"] = out[Q1_COL].map(yn_to01)
    out["Bid_SB"] = out[BID_COL]
    out["Bid_DB"] = np.select(
        [out[Q1_COL].eq("예"), out[Q1_COL].eq("아니오")],
        [out[BID_COL] * 2, out[BID_COL] * 0.5],
        default=np.nan,
    )
    out["Y_DB"] = np.nan
    out.loc[out[Q1_COL].eq("예"), "Y_DB"] = out.loc[out[Q1_COL].eq("예"), Q2_YES_COL].map(yn_to01)
    out.loc[out[Q1_COL].eq("아니오"), "Y_DB"] = out.loc[out[Q1_COL].eq("아니오"), Q2_NO_COL].map(yn_to01)
    return out


def model_frame(frame: pd.DataFrame, boundary: str) -> pd.DataFrame:
    y_col, bid_col = ("Y_SB", "Bid_SB") if boundary == "single" else ("Y_DB", "Bid_DB")
    cols = [y_col, bid_col, *COV_MAP.values()]
    out = frame[cols].rename(columns={y_col: "Y", bid_col: "Bid", **{v: k for k, v in COV_MAP.items()}})
    return out.dropna(subset=["Y", "Bid"])


def make_design(frame: pd.DataFrame, with_covariates: bool, rare_min_count=3):
    """로짓용 y, X와 범주형 변수별 더미 묶음 반환"""
    work = frame.copy()
    X = pd.DataFrame({"Bid": pd.to_numeric(work["Bid"], errors="coerce") / 100_000.0}, index=work.index)
    blocks = {}

    if with_covariates:
        cat_vars = [c for c in ["Gen", "Age", "Edu", "Pos", "Res", "Rti"] if c in work.columns]
        if cat_vars:
            work = collapse_rare_categories(work, cat_vars, min_count=rare_min_count)
            dummies = pd.get_dummies(work[cat_vars].astype("string"), drop_first=True, dummy_na=True).astype(float)
            X = pd.concat([X, dummies], axis=1)
            blocks.update({var: [c for c in dummies.columns if c.startswith(f"{var}_")] for var in cat_vars})

        if "Inf" in work.columns:
            inf = work["Inf"].map(to_numeric_safe)
            if inf.notna().any():
                scale = inf.std(ddof=0)
                X["Inf"] = (inf - inf.mean()) / (scale if scale and np.isfinite(scale) else 1.0)
                blocks["Inf"] = ["Inf"]

    y = pd.to_numeric(work["Y"], errors="coerce")
    valid = y.notna() & X.notna().all(axis=1)
    X, y = X.loc[valid], y.loc[valid]
    X = X[[c for c in X.columns if X[c].nunique(dropna=True) > 1]]
    blocks = {k: [c for c in cols if c in X.columns] for k, cols in blocks.items()}
    blocks = {k: cols for k, cols in blocks.items() if cols}
    return y, X, blocks


def loglike_binomial(y, linpred):
    p = np.clip(1 / (1 + np.exp(-linpred)), 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def null_loglike(y):
    X0 = sm.add_constant(pd.DataFrame(index=y.index), has_constant="add")
    with quiet_statsmodels():
        return float(sm.Logit(y, X0).fit(disp=False, maxiter=400, method="lbfgs").llf)


def hessian_covariance(model, params, ridge_eps=1e-6):
    try:
        with quiet_statsmodels():
            hessian = model.hessian(params)
            return np.linalg.pinv(-hessian + ridge_eps * np.eye(hessian.shape[0]))
    except (LinAlgError, ValueError):
        size = len(params)
        return np.full((size, size), np.nan)


def robust_result(res):
    """statsmodels 버전에 따라 robust covariance 호출 방식이 달라질 수 있어 한 곳에서 처리"""
    try:
        return res.get_robustcov_results(cov_type="HC1")
    except AttributeError:
        return res._get_robustcov_results(cov_type="HC1") or res


def fit_logit_dict(y: pd.Series, X: pd.DataFrame):
    """표준 Logit → GLM → L2 정규화 Logit 순서로 적합"""
    Xc = sm.add_constant(X, has_constant="add")
    if Xc.shape[1] <= 1 or y.nunique() < 2 or len(y) == 0:
        return None

    try:
        with quiet_statsmodels():
            model = sm.Logit(y, Xc)
            res = model.fit(disp=False, maxiter=400, method="lbfgs")
            rob = robust_result(res)
        llf = float(res.llf)
        return {
            "method": "Logit(MLE)",
            "params": rob.params, "bse": rob.bse, "pvalues": rob.pvalues,
            "llf": llf, "lr": 2 * (llf - null_loglike(y)), "aic": float(res.aic),
            "nobs": int(res.nobs), "cols": list(Xc.columns),
        }
    except Exception:
        pass

    try:
        with quiet_statsmodels():
            glm = sm.GLM(y, Xc, family=sm.families.Binomial())
            res = glm.fit(maxiter=400)
            rob = robust_result(res)
        llf = loglike_binomial(y, np.dot(Xc, rob.params))
        return {
            "method": "GLM Binomial",
            "params": rob.params, "bse": rob.bse, "pvalues": rob.pvalues,
            "llf": llf, "lr": 2 * (llf - null_loglike(y)), "aic": -2 * llf + 2 * len(rob.params),
            "nobs": int(res.nobs), "cols": list(Xc.columns),
        }
    except Exception:
        pass

    try:
        with quiet_statsmodels():
            model = sm.Logit(y, Xc)
            res = model.fit_regularized(L1_wt=0.0, alpha=1e-4, maxiter=5000, disp=False, trim_mode="off", cnvrg_tol=1e-10)
    except Exception:
        return None

    params = pd.Series(res.params, index=Xc.columns, dtype="float64")
    se = np.sqrt(np.clip(np.diag(hessian_covariance(model, params)), 0, None))
    pvals = 2 * (1 - norm.cdf(np.abs(params / se)))
    llf = loglike_binomial(y, np.dot(Xc, params))
    try:
        lr = 2 * (llf - null_loglike(y))
    except Exception:
        lr = np.nan
    return {
        "method": "Regularized Logit",
        "params": params,
        "bse": pd.Series(se, index=Xc.columns, dtype="float64"),
        "pvalues": pd.Series(pvals, index=Xc.columns, dtype="float64"),
        "llf": llf, "lr": lr, "aic": -2 * llf + 2 * len(params),
        "nobs": int(len(y)), "cols": list(Xc.columns),
    }


def block_lr_pvalues(y, X, blocks):
    full = fit_logit_dict(y, X)
    if full is None:
        return {}, None
    pvals = {}
    for var, cols in blocks.items():
        reduced = fit_logit_dict(y, X.drop(columns=cols, errors="ignore"))
        if reduced is None or not np.isfinite(full["llf"]) or not np.isfinite(reduced["llf"]):
            pvals[var] = np.nan
            continue
        dof = (len(full["cols"]) - 1) - (len(reduced["cols"]) - 1)
        pvals[var] = 1 - chi2.cdf(2 * (full["llf"] - reduced["llf"]), dof) if dof > 0 else np.nan
    return pvals, full


def summarize_model(y, X, blocks):
    block_pvals, res = block_lr_pvalues(y, X, blocks)
    out = pd.DataFrame(index=MODEL_ROWS, columns=["Coefficient", "StdErr", "p-value"], dtype=float)
    if res is None:
        return out

    params, bse, pvals = res["params"], res["bse"], res["pvalues"]
    out.loc["상수"] = [params.get("const", np.nan), bse.get("const", np.nan), pvals.get("const", np.nan)]
    out.loc["Bid"] = [params.get("Bid", np.nan), bse.get("Bid", np.nan), pvals.get("Bid", np.nan)]

    for var in ["Gen", "Age", "Edu", "Pos", "Res", "Rti"]:
        cols = blocks.get(var, [])
        if cols:
            out.loc[var] = [params.reindex(cols).mean(), bse.reindex(cols).mean(), block_pvals.get(var, np.nan)]
    if "Inf" in blocks:
        out.loc["Inf"] = [params.get("Inf", np.nan), bse.get("Inf", np.nan), pvals.get("Inf", np.nan)]

    out.loc["Log likelihood", "Coefficient"] = res["llf"]
    out.loc["χ²", "Coefficient"] = res["lr"]
    out.loc["Akaike I.C.", "Coefficient"] = res["aic"]
    return out


def combine_first_stage_tables(sb_bid, sb_cov, db_bid, db_cov):
    columns = pd.MultiIndex.from_product(
        [["단일경계", "이중경계"], ["Bid 금액만 포함", "Bid 금액과 공변량 포함"], ["Coefficient", "Standard error", "p-value"]],
        names=["구분", "세부", "통계"],
    )
    final = pd.DataFrame(index=MODEL_ROWS, columns=columns, dtype="float64")

    for block, boundary, spec in [
        (sb_bid, "단일경계", "Bid 금액만 포함"),
        (sb_cov, "단일경계", "Bid 금액과 공변량 포함"),
        (db_bid, "이중경계", "Bid 금액만 포함"),
        (db_cov, "이중경계", "Bid 금액과 공변량 포함"),
    ]:
        final.loc[:, (boundary, spec, "Coefficient")] = pd.to_numeric(block["Coefficient"], errors="coerce").to_numpy()
        final.loc[:, (boundary, spec, "Standard error")] = pd.to_numeric(block["StdErr"], errors="coerce").to_numpy()
        final.loc[:, (boundary, spec, "p-value")] = pd.to_numeric(block["p-value"], errors="coerce").to_numpy()
    return final


def run_first_stage_for_group(group: pd.DataFrame, group_name: str, outdir: Path) -> None:
    single = model_frame(group, "single")
    double = model_frame(group, "double")
    designs = {
        "sb_bid": make_design(single, with_covariates=False),
        "sb_cov": make_design(single, with_covariates=True, rare_min_count=4),
        "db_bid": make_design(double, with_covariates=False),
        "db_cov": make_design(double, with_covariates=True, rare_min_count=4),
    }
    blocks = {name: summarize_model(*design) for name, design in designs.items()}
    final = combine_first_stage_tables(blocks["sb_bid"], blocks["sb_cov"], blocks["db_bid"], blocks["db_cov"]).round(3)

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"WTP_Logit_Table_{group_name}.csv"
    final.to_csv(outpath, encoding="utf-8-sig")
    print(f"[{group_name}] 저장 완료: {outpath.resolve()}")


def prepare_preliminary_wtp_data(df: pd.DataFrame) -> pd.DataFrame:
    """단일경계/이중경계 분석에 필요한 응답·Bid 파생변수 생성"""
    return add_wtp_response_columns(df)


def run_preliminary_first_stage(df: pd.DataFrame, outdir: Path = Path("wtp_logit_outputs")) -> None:
    """
    Info1 그룹별로 1차 WTP 로짓표 저장
    노트북 셀은 `run_preliminary_first_stage(df)`처럼 간단히 호출하고, statsmodels의 수렴/완전분리/헤시안 경고는 이 함수 안에서 조용히 처리
    """
    with suppress_model_warnings():
        for group_name, df_group in df.groupby(INFO_COL):
            run_first_stage_for_group(df_group, str(group_name), outdir)


# 2차 WTP 로짓모형: 1차 full 모형에서 유의한 공변량만 남겨 재적합
SECOND_STAGE_ROWS = [*MODEL_ROWS, "Observations"]


def significant_covariates(result, alpha=0.05):
    if result is None:
        return []
    return [
        col for col, pval in result["pvalues"].items()
        if col not in {"const", "Bid"} and pd.notna(pval) and pval < alpha
    ]


def result_to_second_stage_block(result):
    out = pd.DataFrame(index=SECOND_STAGE_ROWS, columns=["Coefficient", "StdErr", "p-value"], dtype="float64")
    if result is None:
        return out

    row_map = {"const": "상수"}
    for col in result["cols"]:
        row = row_map.get(col, col)
        if row in out.index:
            out.loc[row] = [result["params"].get(col, np.nan), result["bse"].get(col, np.nan), result["pvalues"].get(col, np.nan)]
    out.loc["Log likelihood", "Coefficient"] = result["llf"]
    out.loc["χ²", "Coefficient"] = result["lr"]
    out.loc["Akaike I.C.", "Coefficient"] = result["aic"]
    out.loc["Observations", "Coefficient"] = result["nobs"]
    return out.round(3)


def combine_second_stage_tables(sb_block, db_block):
    columns = pd.MultiIndex.from_product(
        [["단일경계", "이중경계"], ["Bid 금액과 공변량 포함"], ["Coefficient", "Standard error", "p-value"]],
        names=["구분", "세부", "통계"],
    )
    final = pd.DataFrame(index=SECOND_STAGE_ROWS, columns=columns, dtype="float64")
    for block, boundary in [(sb_block, "단일경계"), (db_block, "이중경계")]:
        final.loc[:, (boundary, "Bid 금액과 공변량 포함", "Coefficient")] = block["Coefficient"]
        final.loc[:, (boundary, "Bid 금액과 공변량 포함", "Standard error")] = block["StdErr"]
        final.loc[:, (boundary, "Bid 금액과 공변량 포함", "p-value")] = block["p-value"]
    return final


def fit_selected_model(frame: pd.DataFrame, boundary: str, alpha=0.05):
    data = model_frame(frame, boundary)
    y_full, X_full, _ = make_design(data, with_covariates=True)
    full = fit_logit_dict(y_full, X_full)
    keep = ["Bid", *[col for col in significant_covariates(full, alpha=alpha) if col in X_full.columns]]
    selected = fit_logit_dict(y_full, X_full[keep])
    return result_to_second_stage_block(selected or full)


def run_second_stage_for_group(group: pd.DataFrame, group_name: str, outdir: Path, alpha=0.05):
    final = combine_second_stage_tables(
        fit_selected_model(group, "single", alpha=alpha),
        fit_selected_model(group, "double", alpha=alpha),
    )
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"WTP_Logit_Table_2nd_{group_name}.csv"
    final.to_csv(outpath, encoding="utf-8-sig")
    print(f"[2차분석-{group_name}] 저장 완료: {outpath.resolve()}")


def run_preliminary_second_stage(df: pd.DataFrame, outdir: Path = Path("wtp_logit_outputs"), alpha: float = 0.05) -> None:
    """1차 full 모형에서 유의한 공변량만 남긴 2차 로짓표 저장"""
    for group_name, df_group in df.groupby(INFO_COL):
        run_second_stage_for_group(df_group, str(group_name), outdir, alpha=alpha)


# 유의 공변량만 활용한 단일/이중경계 로짓표 저장
NUMERIC_COVS = list(COV_MAP.keys())
SECOND_STAGE_BASE_INDEX = ["const", "Bid", *NUMERIC_COVS]
SECOND_STAGE_OUTPUT_INDEX = ["상수", "Bid", *NUMERIC_COVS, "Log likelihood", "χ²", "Akaike I.C.", "Observations"]


def prepare_preliminary_numeric_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """표본 WTP 추정 단계에서 쓰는 숫자형 공변량과 SB/DB 파생변수 정리"""
    df = df.copy()
    df["Y_SB"] = df[Q1_COL].map(yn_to01)
    df["Bid_SB"] = pd.to_numeric(df[BID_COL], errors="coerce")
    df["Bid_DB"] = np.select([df[Q1_COL].eq("예"), df[Q1_COL].eq("아니오")], [2 * df["Bid_SB"], 0.5 * df["Bid_SB"]], default=np.nan)
    df["Y_DB"] = np.nan
    df.loc[df[Q1_COL].eq("예"), "Y_DB"] = df.loc[df[Q1_COL].eq("예"), Q2_YES_COL].map(yn_to01)
    df.loc[df[Q1_COL].eq("아니오"), "Y_DB"] = df.loc[df[Q1_COL].eq("아니오"), Q2_NO_COL].map(yn_to01)
    for short, original in COV_MAP.items():
        df[short] = pd.to_numeric(df[original], errors="coerce")
    return df


def numeric_design(frame: pd.DataFrame, y_col: str, bid_col: str):
    y = pd.to_numeric(frame[y_col], errors="coerce")
    X = pd.DataFrame({"Bid": pd.to_numeric(frame[bid_col], errors="coerce")}, index=frame.index)
    for cov in NUMERIC_COVS:
        X[cov] = pd.to_numeric(frame[cov], errors="coerce")
    return y, X


def fit_logit_result(y: pd.Series, X: pd.DataFrame):
    Xc = sm.add_constant(X, has_constant="add")
    valid = y.notna() & Xc.notna().all(axis=1)
    Xc, yv = Xc.loc[valid], y.loc[valid].astype(float)
    if yv.nunique() < 2 or Xc.shape[0] <= Xc.shape[1] + 1:
        return None
    try:
        with quiet_statsmodels():
            return sm.Logit(yv, Xc).fit(disp=False, maxiter=300)
    except Exception:
        pass

    try:
        with quiet_statsmodels():
            reg = sm.Logit(yv, Xc).fit_regularized(L1_wt=0.0, alpha=1e-4, maxiter=5000, trim_mode="off", cnvrg_tol=1e-10, disp=False)
    except Exception:
        return None

    params = pd.Series(reg.params, index=Xc.columns)
    cov = hessian_covariance(sm.Logit(yv, Xc), params)
    bse = pd.Series(np.sqrt(np.clip(np.diag(cov), 0, None)), index=Xc.columns)

    class ResultLike:
        def __init__(self, params, bse, cov):
            self.params = params
            self.bse = bse
            self._cov = pd.DataFrame(cov, index=params.index, columns=params.index)
            self.pvalues = pd.Series(2 * (1 - norm.cdf(np.abs(params / bse))), index=params.index)
            self.llf = loglike_binomial(yv, np.dot(Xc, params))
            self.aic = 2 * len(params) - 2 * self.llf
            self.nobs = Xc.shape[0]

        def cov_params(self):
            return self._cov

    return ResultLike(params, bse, cov)


def selected_numeric_result(frame: pd.DataFrame, y_col: str, bid_col: str, alpha=0.05):
    y, X = numeric_design(frame, y_col, bid_col)
    full = fit_logit_result(y, X)
    if full is not None:
        selected = [c for c in NUMERIC_COVS if c in full.pvalues.index and pd.notna(full.pvalues[c]) and full.pvalues[c] < alpha]
        for keep in [["Bid", *selected], ["Bid", "Inf"], ["Bid"]]:
            keep = [c for c in keep if c in X.columns]
            result = fit_logit_result(y, X[keep])
            if result is not None:
                return result
    return fit_logit_result(y, X[["Bid", "Inf"]]) or fit_logit_result(y, X[["Bid"]])


def result_to_numeric_block(result):
    block = pd.DataFrame(index=SECOND_STAGE_BASE_INDEX, columns=["Coefficient", "Standard error", "p-value"], dtype="float64")
    if result is not None:
        for row in block.index:
            block.loc[row] = [result.params.get(row, np.nan), result.bse.get(row, np.nan), result.pvalues.get(row, np.nan)]
        stats_block = pd.DataFrame({
            "Coefficient": [getattr(result, "llf", np.nan), np.nan, getattr(result, "aic", np.nan), getattr(result, "nobs", np.nan)],
            "Standard error": [np.nan] * 4,
            "p-value": [np.nan] * 4,
        }, index=["Log likelihood", "χ²", "Akaike I.C.", "Observations"])
    else:
        stats_block = pd.DataFrame(index=["Log likelihood", "χ²", "Akaike I.C.", "Observations"], columns=block.columns, dtype="float64")
    return pd.concat([block, stats_block]).rename(index={"const": "상수"})


def combine_numeric_blocks(sb_block, db_block):
    columns = pd.MultiIndex.from_product(
        [["단일경계", "이중경계"], ["Bid 금액과 공변량 포함"], ["Coefficient", "Standard error", "p-value"]],
        names=["구분", "세부", "지표"],
    )
    final = pd.DataFrame(index=SECOND_STAGE_OUTPUT_INDEX, columns=columns, dtype=float)
    for block, boundary in [(sb_block, "단일경계"), (db_block, "이중경계")]:
        final.loc[block.index, (boundary, "Bid 금액과 공변량 포함", "Coefficient")] = block["Coefficient"].values
        final.loc[block.index, (boundary, "Bid 금액과 공변량 포함", "Standard error")] = block["Standard error"].values
        final.loc[block.index, (boundary, "Bid 금액과 공변량 포함", "p-value")] = block["p-value"].values
    return final.round(3)


def run_preliminary_numeric_second_stage(df: pd.DataFrame, outdir: Path = Path("wtp_logit_outputs"), alpha: float = 0.05) -> None:
    """유의 공변량만 활용한 단일/이중경계 로짓표 저장"""
    outdir.mkdir(parents=True, exist_ok=True)
    for group_name, df_group in df.groupby(INFO_COL):
        final = combine_numeric_blocks(
            result_to_numeric_block(selected_numeric_result(df_group, "Y_SB", "Bid_SB", alpha=alpha)),
            result_to_numeric_block(selected_numeric_result(df_group, "Y_DB", "Bid_DB", alpha=alpha)),
        )
        outpath = outdir / f"WTP_SecondStage_{group_name}.csv"
        final.to_csv(outpath, encoding="utf-8-sig")
        print(f"[{group_name}] 저장 완료: {outpath}")


# 2차 추정치를 활용한 표본 WTP 요약표
ALPHA = 0.05
USE_OBSERVED_AMAX = True
AMAX_FALLBACK = 1_200_000
WTP_COLUMNS = [
    "구분", "경계",
    "WTP평균", "표준오차", "95% 신뢰구간 (하한, 상한)",
    "WTP중앙값", "표준오차.1", "95% 신뢰구간 (하한, 상한).1",
    "WTP절단된평균값", "표준오차.2", "95% 신뢰구간 (하한, 상한).2",
]


def round3(value):
    return np.round(value, 3) if pd.notna(value) and np.isfinite(value) else np.nan


def ci95(estimate, se):
    if not np.isfinite(se):
        return np.nan, np.nan
    return estimate - 1.96 * se, estimate + 1.96 * se


def delta_se(gradient, covariance):
    if covariance is None or not np.isfinite(covariance).all():
        return np.nan
    return float(np.sqrt(max(0.0, gradient @ covariance @ gradient)))


def alpha_beta_covariance(result, X_used, used_covs):
    if result is None or "Bid" not in result.params.index:
        return np.nan, np.nan, None

    params = result.params
    cov = result.cov_params()
    if isinstance(cov, np.ndarray):
        cov = pd.DataFrame(cov, index=params.index, columns=params.index)

    means = X_used[used_covs].mean() if used_covs else pd.Series(dtype=float)
    alpha_vector = pd.Series(0.0, index=params.index)
    alpha_vector.loc["const"] = 1.0 if "const" in alpha_vector.index else 0.0
    for cov_name in used_covs:
        if cov_name in alpha_vector.index:
            alpha_vector[cov_name] = float(means.get(cov_name, 0.0))

    alpha = float(alpha_vector @ params)
    beta = -float(params["Bid"])
    covariance = np.array([
        [float(alpha_vector @ cov.loc[alpha_vector.index, alpha_vector.index] @ alpha_vector), -float(alpha_vector @ cov.loc[alpha_vector.index, "Bid"])],
        [-float(alpha_vector @ cov.loc[alpha_vector.index, "Bid"]), float(cov.loc["Bid", "Bid"])],
    ])
    return alpha, beta, covariance


def wtp_statistics(alpha, beta, amax, covariance):
    mean = np.log1p(np.exp(alpha)) / beta
    median = alpha / beta
    trunc = (np.log1p(np.exp(alpha)) - np.log1p(np.exp(alpha - beta * amax))) / beta

    sigmoid_alpha = 1 / (1 + np.exp(-alpha))
    sigmoid_trunc = 1 / (1 + np.exp(-(alpha - beta * amax)))
    gradients = {
        "mean": np.array([sigmoid_alpha / beta, -np.log1p(np.exp(alpha)) / beta**2]),
        "median": np.array([1 / beta, -alpha / beta**2]),
        "trunc": np.array([(sigmoid_alpha - sigmoid_trunc) / beta, -trunc / beta + (amax * sigmoid_trunc) / beta]),
    }
    ses = {key: delta_se(gradient, covariance) for key, gradient in gradients.items()}
    return {"mean": (mean, ses["mean"]), "median": (median, ses["median"]), "trunc": (trunc, ses["trunc"])}


def estimate_wtp_block(frame: pd.DataFrame, y_col: str, bid_col: str):
    y, X = numeric_design(frame, y_col, bid_col)
    full = fit_logit_result(y, X)
    used_covs = [] if full is None else [c for c in NUMERIC_COVS if c in full.pvalues.index and pd.notna(full.pvalues[c]) and full.pvalues[c] < ALPHA]
    X_used = X[["Bid", *used_covs]] if used_covs else X[["Bid"]]
    result = fit_logit_result(y, X_used) or full

    alpha, beta, covariance = alpha_beta_covariance(result, X_used, used_covs)
    amax = float(np.nanmax(pd.to_numeric(frame[bid_col], errors="coerce"))) if USE_OBSERVED_AMAX else AMAX_FALLBACK
    stats = wtp_statistics(alpha, beta, amax, covariance)

    def formatted_ci(estimate, se):
        lo, hi = ci95(estimate, se)
        return f"({round3(lo)}, {round3(hi)})"

    return {
        "WTP평균": round3(stats["mean"][0]),
        "표준오차(평균)": round3(stats["mean"][1]),
        "95% 신뢰구간(평균)": formatted_ci(*stats["mean"]),
        "WTP중앙값": round3(stats["median"][0]),
        "표준오차(중앙값)": round3(stats["median"][1]),
        "95% 신뢰구간(중앙)": formatted_ci(*stats["median"]),
        "WTP절단된평균값": round3(stats["trunc"][0]),
        "표준오차(절단)": round3(stats["trunc"][1]),
        "95% 신뢰구간(절단)": formatted_ci(*stats["trunc"]),
    }


def group_label(value):
    return "다른 정보플랫폼 정보이용료 정보가 포함된 것" if str(value).strip() in {"정보O", "정보o", "O", "o", "Yes", "1"} else "다른 정보플랫폼 정보이용료 정보가 포함되지 않은 것"


def wtp_row(group_name, boundary, estimates):
    return {
        "구분": group_label(group_name),
        "경계": boundary,
        "WTP평균": estimates["WTP평균"],
        "표준오차": estimates["표준오차(평균)"],
        "95% 신뢰구간 (하한, 상한)": estimates["95% 신뢰구간(평균)"],
        "WTP중앙값": estimates["WTP중앙값"],
        "표준오차.1": estimates["표준오차(중앙값)"],
        "95% 신뢰구간 (하한, 상한).1": estimates["95% 신뢰구간(중앙)"],
        "WTP절단된평균값": estimates["WTP절단된평균값"],
        "표준오차.2": estimates["표준오차(절단)"],
        "95% 신뢰구간 (하한, 상한).2": estimates["95% 신뢰구간(절단)"],
    }


def run_preliminary_wtp_summary(df: pd.DataFrame, outdir: Path = Path("wtp_from_second_stage")) -> None:
    """2차 추정치를 활용해 그룹별 표본 WTP 요약표를 저장하고 표시"""
    rows = []
    for group_name, df_group in df.groupby(INFO_COL):
        rows.append(wtp_row(group_name, "단일경계", estimate_wtp_block(df_group, "Y_SB", "Bid_SB")))
        rows.append(wtp_row(group_name, "이중경계", estimate_wtp_block(df_group, "Y_DB", "Bid_DB")))

    order = [
        ("다른 정보플랫폼 정보이용료 정보가 포함된 것", "단일경계"),
        ("다른 정보플랫폼 정보이용료 정보가 포함된 것", "이중경계"),
        ("다른 정보플랫폼 정보이용료 정보가 포함되지 않은 것", "단일경계"),
        ("다른 정보플랫폼 정보이용료 정보가 포함되지 않은 것", "이중경계"),
    ]
    final = pd.DataFrame(rows, columns=WTP_COLUMNS)
    final["__ord__"] = final.apply(lambda row: order.index((row["구분"], row["경계"])) if (row["구분"], row["경계"]) in order else 999, axis=1)
    final = final.sort_values("__ord__").drop(columns="__ord__").reset_index(drop=True)

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "WTP_표본추정_요약_최종.csv"
    final.to_csv(outpath, index=False, encoding="utf-8-sig")
    display(final)
    print(f"저장 완료: {outpath}")


# 모델 평가용 RMSE: 단일/이중경계 x Bid-only/Bid+공변량
RMSE_COLUMNS = [
    "정보이용료 포함 여부",
    "단일경계: Bid만",
    "단일경계: Bid+공변량",
    "이중경계: Bid만",
    "이중경계: Bid+공변량",
]


def build_rmse_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    parts = [pd.to_numeric(frame[COV_MAP["Inf"]], errors="coerce").rename("Inf")]

    age_num = pd.to_numeric(frame[COV_MAP["Age"]], errors="coerce")
    use_age_numeric = age_num.notna().sum() >= 3 and age_num.nunique(dropna=True) >= 2
    if use_age_numeric:
        parts.append(age_num.rename("Age"))

    nominal_vars = ["Gen", "Edu", "Pos", "Res", "Rti"] if use_age_numeric else ["Gen", "Age", "Edu", "Pos", "Res", "Rti"]
    for key in nominal_vars:
        parts.append(pd.get_dummies(frame[COV_MAP[key]].astype("category"), prefix=key, drop_first=True))

    covariates = pd.concat(parts, axis=1)
    keep = [col for col in covariates.columns if pd.to_numeric(covariates[col], errors="coerce").var(skipna=True) > 0]
    return covariates[keep]


def constant_rmse(y: pd.Series) -> float:
    yv = pd.to_numeric(y, errors="coerce").dropna().astype(float)
    if len(yv) == 0:
        return np.nan
    p = (yv.sum() + 0.5) / (len(yv) + 1.0)
    return float(np.sqrt(np.mean((yv - p) ** 2)))


def safe_logit_rmse(y: pd.Series, X: pd.DataFrame) -> float:
    data = pd.concat([y.rename("Y"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data["Y"].nunique() < 2:
        return constant_rmse(y)

    yv = data["Y"].astype(float)
    Xc = sm.add_constant(data.drop(columns="Y").astype(float), has_constant="add")
    for kwargs in [{"disp": False, "maxiter": 200}, {"alpha": 1e-5, "L1_wt": 0.0, "maxiter": 5000, "cnvrg_tol": 1e-10, "disp": False}]:
        try:
            with quiet_statsmodels():
                res = sm.Logit(yv, Xc).fit(**kwargs) if "alpha" not in kwargs else sm.Logit(yv, Xc).fit_regularized(**kwargs)
            pred = res.predict(Xc)
            return float(np.sqrt(np.mean((yv - pred) ** 2)))
        except Exception:
            continue
    return constant_rmse(yv)


def rmse_specs_for_group(frame: pd.DataFrame):
    y_sb = pd.to_numeric(frame["Y_SB"], errors="coerce")
    y_db = pd.to_numeric(frame["Y_DB"], errors="coerce")
    X_sb_bid = pd.DataFrame({"Bid": pd.to_numeric(frame["Bid_SB"], errors="coerce")}, index=frame.index)
    X_db_bid = pd.DataFrame({"Bid": pd.to_numeric(frame["Bid_DB"], errors="coerce")}, index=frame.index)
    covariates = build_rmse_covariates(frame)
    return [
        safe_logit_rmse(y_sb, X_sb_bid),
        safe_logit_rmse(y_sb, pd.concat([X_sb_bid, covariates], axis=1)),
        safe_logit_rmse(y_db, X_db_bid),
        safe_logit_rmse(y_db, pd.concat([X_db_bid, covariates], axis=1)),
    ]


def run_preliminary_rmse(df: pd.DataFrame, outdir: Path = Path("wtp_logit_outputs")) -> None:
    """단일/이중경계와 Bid-only/Bid+공변량 4개 사양의 RMSE 저장"""
    rows = [[str(group_name), *rmse_specs_for_group(df_group)] for group_name, df_group in df.groupby(INFO_COL)]
    rmse_tbl = pd.DataFrame(rows, columns=RMSE_COLUMNS).sort_values("정보이용료 포함 여부")
    rmse_tbl.iloc[:, 1:] = rmse_tbl.iloc[:, 1:].astype(float).round(4)

    outdir.mkdir(parents=True, exist_ok=True)
    outf = outdir / "RMSE_Logit_4spec_noNaN.csv"
    rmse_tbl.to_csv(outf, index=False, encoding="utf-8-sig")

    print(rmse_tbl)
    print(f"[저장 완료] {outf}")
