"""한국어 텍스트 탐구실. 입력 자료와 결과는 세션 메모리에서만 처리한다."""
from collections import Counter
from io import BytesIO
from pathlib import Path
import re
import base64

import streamlit as st
import pandas as pd
import numpy as np
from kiwipiepy import Kiwi, Match
from wordcloud import WordCloud
from PIL import Image, UnidentifiedImageError
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

FONT = Path(__file__).parent / "fonts" / "NanumGothic.ttf"
STOP = "있다, 하다, 되다, 것, 수, 등, 이다, 않다, 없다"
TAGS = {"NNG": "일반명사", "NNP": "고유명사", "VV": "동사", "VA": "형용사", "MAG": "부사", "SL": "외국어"}
EXAMPLES = {
    "영화 댓글": "이 영화는 긴장감과 감동이 있는 영화예요.\n\n이 영화는 감동은 있지만 지루해서 실망이에요.\n\n추천! 눈물 없이는 볼 수 없는 감동적인 영화입니다.",
    "청바지 후기": "청바지 추천! 예쁘고 저렴해서 청바지를 추천해요.\n\n청바지의 가격, 디자인 다 좋아요. 특히 디자인 추천!^^\n\n신축성과 디자인이 좋고, 가격도 좋아서 이 청바지를 추천해요.",
}


@st.cache_resource(scope="session", show_spinner="한국어 분석기를 준비하고 있습니다…")
def get_kiwi():
    # 세션마다 한 번만 생성. 사용자 사전이 다른 이용자에게 전파되지 않는다.
    return Kiwi(num_workers=1, model_type="cong"), set()


def split_words(value):
    return {w.strip() for w in re.split(r"[,\n]", value) if w.strip()}


def decode_text(data):
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError("파일을 읽지 못했습니다. UTF-8 또는 CP949 형식의 텍스트 파일로 저장해 주세요.")


def analyze(kiwi, text, divide, tags, stops, single):
    docs = [p.strip() for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if p.strip()] if divide else [text]
    counts, raw = [], []
    for i, doc in enumerate(docs):
        count = Counter()
        for token in kiwi.tokenize(doc, match_options=Match.ALL | Match.JOIN_ADJ_SUFFIX):
            tag = token.tag.split("-")[0]
            word = token.form + "다" if tag in ("VV", "VA") else token.form
            keep = tag in tags and word not in stops and not (single and len(word) == 1)
            raw.append({"문서": f"문서 {i+1}", "원문 조각": doc[token.start:token.start+token.len], "형태소": token.form, "품사": token.tag, "표시 단어": word, "분석에 포함": keep})
            if keep:
                count[(word, tag)] += 1
        counts.append(count)
    total = sum(counts, Counter())
    keys = sorted(total, key=lambda k: (-total[k], k))
    matrix = pd.DataFrame([[c[k] for k in keys] for c in counts], columns=range(len(keys)), index=[f"문서 {i+1}" for i in range(len(docs))])
    size = sum(total.values())
    table = pd.DataFrame([{"포함": True, "단어": w, "품사": TAGS[t], "빈도": total[(w,t)], "상대도수": total[(w,t)] / size} for w,t in keys], columns=["포함", "단어", "품사", "빈도", "상대도수"])
    return table, matrix, pd.DataFrame(raw)


def merged_matrix(table, matrix):
    result = {}
    for idx, row in table.iterrows():
        name = str(row["단어"]).strip() if pd.notna(row["단어"]) else ""
        if row["포함"] and name:
            result[name] = result.get(name, pd.Series(0, index=matrix.index)) + matrix[idx]
    return pd.DataFrame(result, index=matrix.index).astype(int)


def tfidf(tf, logarithm):
    df = (tf > 0).sum(axis=0)
    idf = len(tf) / df
    if logarithm:
        idf = np.log(idf)
    return df, idf, tf.mul(idf, axis=1)


def save_editor():
    table = st.session_state.table.copy()
    for idx, edits in st.session_state.editor.get("edited_rows", {}).items():
        for column, value in edits.items():
            table.at[int(idx), column] = value
    st.session_state.table = table
    # 새 편집기 기준표로 교체해 편집 델타가 중복 적용되지 않게 한다.
    del st.session_state["editor"]


def set_inclusion(top=None):
    table = st.session_state.table.copy()
    names = table["단어"].fillna("").str.strip()
    sums = table.assign(단어=names).groupby("단어")["빈도"].sum().sort_values(ascending=False)
    chosen = set(sums.head(top).index) if top else set(sums.index)
    table["포함"] = names.isin(chosen) & names.ne("")
    st.session_state.table = table
    st.session_state.pop("editor", None)


def load_example():
    name = st.session_state.example
    st.session_state.text = EXAMPLES[name]
    st.session_state.divide = name == "청바지 후기"
    for tag in TAGS:
        st.session_state[f"tag_{tag}"] = tag in ({"NNG", "NNP", "VV", "VA"} if name == "영화 댓글" else {"NNG", "NNP", "VA"})
    st.session_state.stops = STOP
    st.session_state.single = False


def shape_mask(shape, upload):
    if shape == "사각형":
        return None
    if shape == "마스크 이미지 업로드":
        if upload is None:
            raise ValueError("모양으로 사용할 PNG 이미지를 올려 주세요. 어두운 영역에 단어가 들어갑니다.")
        image = Image.open(BytesIO(upload.getvalue())).convert("RGBA")
        image.thumbnail((1000, 1000))
        backdrop = Image.new("RGBA", image.size, "white")
        backdrop.alpha_composite(image)
        gray = np.asarray(backdrop.convert("L"))
        mask = np.where(gray < 128, 0, 255).astype(np.uint8)
    else:
        y, x = np.mgrid[-1.3:1.3:700j, -1.3:1.3:700j]
        inside = x*x + y*y <= 1 if shape == "원" else (x*x + (-y)**2 - 1)**3 - x*x*(-y)**3 <= 0
        mask = np.where(inside, 0, 255).astype(np.uint8)
    if (mask == 0).sum() < 100:
        raise ValueError("단어가 들어갈 공간이 너무 작습니다. 어두운 영역이 넓은 이미지를 사용해 주세요.")
    return mask


def png_bytes(image):
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def main():
    st.set_page_config(page_title="우리말 데이터 탐구실", page_icon="🌿", layout="wide")
    css = ""
    if FONT.exists():
        encoded = base64.b64encode(FONT.read_bytes()).decode()
        css = f"@font-face {{font-family: Nanum; src: url(data:font/ttf;base64,{encoded});}} html, body, [class*='st-'], h1,h2,h3,p {{font-family: Nanum, sans-serif;}}"
    st.markdown(f"<style>{css} .stMainBlockContainer{{max-width:1200px;}} p,label,[data-testid='stWidgetLabel'] p{{font-size:18px!important;}} h2{{color:#226558;}} .stButton button{{min-height:44px;}} </style>", unsafe_allow_html=True)
    st.markdown("""<style>
    [data-testid="stFileUploaderDropzoneInstructions"] {font-size:0;}
    [data-testid="stFileUploaderDropzoneInstructions"] > * {display:none;}
    [data-testid="stFileUploaderDropzoneInstructions"]::after {content:"파일을 여기에 놓으세요 · 최대 5메가바이트";font-size:16px;}
    [data-testid="stFileUploaderDropzone"] button {font-size:0;}
    [data-testid="stFileUploaderDropzone"] button::after {content:"파일 선택";font-size:16px;}
    </style>""", unsafe_allow_html=True)
    st.title("🌿 우리말 데이터 탐구실")
    st.write("초등수학영재프로그램개발 · 자연어 처리 → 단어 빈도 → 워드클라우드")
    st.caption("입력한 글과 분석 결과는 현재 세션의 메모리에서만 사용하며 파일이나 데이터베이스에 저장하지 않습니다.")
    if not FONT.exists():
        st.warning("한글 글꼴이 없습니다. 앱 폴더의 fonts/NanumGothic.ttf에 나눔고딕을 넣어 주세요. 표 분석은 가능하며 그림 생성은 글꼴을 넣은 뒤 사용할 수 있습니다.")
    st.session_state.setdefault("text", "")
    for tag in TAGS:
        st.session_state.setdefault(f"tag_{tag}", tag in ("NNG", "NNP"))
    st.session_state.setdefault("stops", STOP)
    with st.sidebar:
        st.header("분석 옵션")
        st.caption("옵션을 바꾼 뒤 ‘분석하기’를 누르면 새 분석을 시작합니다.")
        tags = {tag for tag,label in TAGS.items() if st.checkbox(f"{label} ({tag})", key=f"tag_{tag}")}
        single = st.checkbox("한 글자 단어 제외", key="single")
        stops = st.text_area("불용어 목록 · 쉼표로 구분", key="stops")
        user_words = st.text_area("사용자 사전 · 쉼표로 구분", placeholder="워드클라우드, 생성형AI")
        st.caption("등록 단어는 고유명사로 처리합니다. 같은 세션에서는 추가 등록이 유지됩니다. 사전을 완전히 초기화하려면 브라우저를 새로고침하세요.")
    st.header("① 텍스트 입력")
    st.write("분석할 글을 붙여 넣거나 텍스트 파일을 불러오세요.")
    st.selectbox("검수용 예시", list(EXAMPLES), key="example")
    st.button("예시 불러오기", on_click=load_example)
    upload = st.file_uploader("텍스트 파일 올리기 · UTF-8 또는 CP949", type=["txt"])
    if upload and st.button("파일 내용을 입력창으로 가져오기"):
        if upload.size > 2_000_000:
            st.error("2메가바이트 이하의 텍스트 파일을 사용해 주세요.")
        else:
            try:
                st.session_state.text = decode_text(upload.getvalue())
            except ValueError as e:
                st.error(str(e))
    text = st.text_area("분석할 텍스트", height=260, key="text", placeholder="여기에 댓글, 후기, 수업 소감 등을 붙여 넣으세요.")
    divide = st.checkbox("문서 나누기 · 빈 줄마다 다른 문서로 분석", key="divide")
    signature = (text, divide, tuple(sorted(tags)), stops, single, user_words)
    if st.button("분석하기", type="primary"):
        if not text.strip() or not tags:
            st.error("텍스트를 입력하고 포함할 품사를 하나 이상 선택해 주세요.")
        elif len(text) > 100_000:
            st.error("수업용 분석은 한 번에 10만 글자까지 가능합니다. 글을 나누어 입력해 주세요.")
        else:
            kiwi, registered = get_kiwi()
            for word in split_words(user_words) - registered:
                kiwi.add_user_word(word, "NNP", 10.0)
                registered.add(word)
            with st.spinner("형태소와 단어 빈도를 계산하고 있습니다…"):
                table, matrix, raw = analyze(kiwi, text, divide, tags, split_words(stops), single)
            st.session_state.update(table=table, matrix=matrix, raw=raw, signature=signature, analyzed_divide=divide)
            st.session_state.pop("editor", None)
    st.header("② 형태소 분석과 빈도표")
    st.write("형태소 = 뜻을 가진 가장 작은 말의 단위. 동사와 형용사는 기본형으로 표시합니다.")
    if "table" not in st.session_state:
        st.info("①에서 글을 입력하고 ‘분석하기’를 눌러 주세요.")
        st.header("③ 시각화")
        st.write("단어의 빈도를 막대의 길이와 글자의 크기로 비교합니다.")
        return
    if signature != st.session_state.signature:
        st.info("입력 또는 분석 옵션이 바뀌었습니다. 현재는 이전 분석 결과입니다. ‘분석하기’를 누르면 표 편집 내용이 초기화됩니다.")
    table = st.session_state.table
    with st.expander("실제 형태소 분리 결과 보기"):
        st.dataframe(st.session_state.raw, hide_index=True)
    if table.empty:
        st.warning("조건에 맞는 단어가 없습니다. 포함할 품사와 불용어를 확인해 주세요.")
        return
    tf = merged_matrix(table, st.session_state.matrix)
    totals = tf.sum().sort_values(ascending=False)
    a,b,c = st.columns(3)
    a.metric("전체 단어 수 · 분석 조건 적용 후", int(table["빈도"].sum()))
    b.metric("서로 다른 단어 수 · 현재 포함", len(totals))
    c.metric("제외한 단어 수 · 표의 행 기준", int((~table["포함"] | table["단어"].fillna("").str.strip().eq("")).sum()))
    st.caption("포함을 끄거나 이름을 고치면 아래 결과가 즉시 바뀝니다. 같은 이름은 결과에서 합산합니다. 상대도수의 분모는 분석 조건 적용 후 전체 단어 수입니다. 열 제목을 눌러 정렬할 수 있습니다.")
    st.data_editor(table, key="editor", on_change=save_editor, hide_index=True, num_rows="fixed", disabled=["품사", "빈도", "상대도수"], column_config={"포함": st.column_config.CheckboxColumn("포함 ✔"), "단어": st.column_config.TextColumn(required=True), "상대도수": st.column_config.NumberColumn(format="%.4f")}, width="stretch")
    n = st.number_input("상위 몇 개 단어를 포함할까요?", min_value=1, max_value=max(1,len(table)), value=min(20,len(table)))
    a,b = st.columns(2)
    a.button("모두 포함", on_click=set_inclusion)
    b.button("상위 N개만 포함", on_click=set_inclusion, args=(int(n),))
    result = pd.DataFrame({"단어":totals.index, "빈도":totals.values})
    result["상대도수"] = result["빈도"] / table["빈도"].sum()
    st.caption("합산 결과 · 내려받기에는 이 결과와 편집표가 사용됩니다.")
    st.dataframe(result, hide_index=True)
    weighted = None
    if st.session_state.analyzed_divide and not tf.empty:
        st.subheader("문서별 TF·IDF 탐구")
        method = st.radio("IDF 계산 방식", ["교과서식 N/DF", "자연로그 log(N/DF)"], horizontal=True)
        df,idf,weighted = tfidf(tf, method.startswith("자연로그"))
        st.caption("TF는 문서 안의 횟수, DF는 등장 문서 수, N은 전체 문서 수입니다. 빈 문단은 문서에서 제외합니다. 제외·이름 합치기는 모든 표에 반영됩니다.")
        st.write("문서 × 단어 TF 표")
        st.dataframe(tf)
        st.write("단어별 DF와 IDF")
        st.dataframe(pd.DataFrame({"DF · 등장 문서 수":df, "IDF":idf}))
        st.write("문서 × 단어 TF·IDF 표")
        st.dataframe(weighted.style.highlight_max(axis=1, props="background-color: #d7eee3; color: #153d30").format("{:.3f}"))
        st.caption("각 문서의 최댓값을 모두 강조합니다. 로그식에서는 모든 문서에 등장하는 단어의 IDF가 0입니다.")
    excel = BytesIO()
    with pd.ExcelWriter(excel, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="합산 빈도", index=False)
        table.to_excel(writer, sheet_name="편집표", index=False)
        if weighted is not None:
            tf.to_excel(writer, sheet_name="문서별 빈도")
            pd.DataFrame({"DF":df,"IDF":idf}).to_excel(writer, sheet_name="문서빈도와 역문서빈도")
            weighted.to_excel(writer, sheet_name="문서별 가중치")
        # 사용자가 입력한 단어가 엑셀 수식으로 실행되지 않도록 문자열로 저장한다.
        for sheet in writer.book:
            for row in sheet:
                for cell in row:
                    if cell.data_type == "f":
                        cell.data_type = "s"
    safe_csv = result.copy()
    safe_csv["단어"] = safe_csv["단어"].map(lambda w: "'"+w if w.startswith(("=", "+", "-", "@", "\t", "\r")) else w)
    a,b = st.columns(2)
    a.download_button("CSV 내려받기", safe_csv.to_csv(index=False).encode("utf-8-sig"), "단어빈도.csv", "text/csv")
    b.download_button("엑셀 내려받기", excel.getvalue(), "단어분석.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.header("③ 시각화")
    st.write("자주 쓰인 단어와 특정 문서에 특징적인 단어를 그림으로 비교합니다.")
    if totals.empty:
        st.info("포함된 단어가 없습니다. 표에서 하나 이상 포함해 주세요.")
        return
    if not FONT.exists():
        return
    chart_n = st.slider("막대그래프 상위 단어 수", 1, max(2,len(totals)), min(15,len(totals)))
    shown = totals.head(chart_n).sort_values()
    font = FontProperties(fname=str(FONT), size=14)
    fig = Figure(figsize=(10,max(4,len(shown)*0.38)), layout="constrained")
    ax = fig.subplots()
    ax.barh(shown.index, shown.values, color="#368574")
    ax.set_xlabel("단어 빈도", fontproperties=font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig)
    chart = BytesIO()
    fig.savefig(chart, format="png", dpi=160)
    st.download_button("막대그래프 PNG 내려받기", chart.getvalue(), "막대그래프.png", "image/png")
    st.subheader("나만의 워드클라우드")
    a,b,c = st.columns(3)
    basis = a.selectbox("크기 기준", ["빈도", "TF·IDF"] if weighted is not None else ["빈도"])
    max_words = b.slider("최대 단어 수",10,200,100)
    background = c.selectbox("배경색",["흰색","검정","투명"])
    a,b = st.columns(2)
    min_size = a.slider("최소 글자 크기",8,80,12)
    max_size = b.slider("최대 글자 크기",min_size,240,max(120,min_size))
    palettes = {"초록과 파랑":"viridis", "노랑과 보라":"plasma", "따뜻한 불꽃":"inferno", "바다":"ocean", "숲":"summer", "다채로운 색":"tab10", "단색":None}
    a,b = st.columns(2)
    palette = a.selectbox("색상 팔레트", list(palettes))
    color = b.color_picker("단색 글자 색", "#287c6b") if palette == "단색" else None
    shape = st.selectbox("모양",["사각형","원","하트","마스크 이미지 업로드"])
    mask_upload = st.file_uploader("마스크 PNG · 어두운 곳에 단어, 흰색·투명 영역은 빈 공간",type=["png"]) if shape == "마스크 이미지 업로드" else None
    a,b = st.columns(2)
    horizontal = a.slider("가로 단어 비율",0.0,1.0,0.9,0.05)
    scaling = b.slider("크기 차이 강조",0.0,1.0,0.5,0.05)
    st.session_state.setdefault("seed",42)
    if st.button("다시 배치"):
        st.session_state.seed += 1
    weights = totals
    if basis == "TF·IDF":
        doc = st.selectbox("가중치를 사용할 문서",["전체 문서 합계",*weighted.index])
        weights = weighted.sum() if doc == "전체 문서 합계" else weighted.loc[doc]
    weights = weights[weights>0]
    if weights.empty:
        st.info("양수인 가중치가 없습니다. 빈도 또는 교과서식 IDF를 선택해 주세요.")
        return
    try:
        mask = shape_mask(shape,mask_upload)
        wc = WordCloud(font_path=str(FONT), width=1100, height=650, mask=mask, background_color={"흰색":"white","검정":"black","투명":None}[background], mode="RGBA", max_words=max_words, min_font_size=min_size, max_font_size=max_size, prefer_horizontal=horizontal, relative_scaling=scaling, random_state=st.session_state.seed, colormap=palettes[palette] or "viridis", color_func=(lambda *args,**kwargs:color) if color else None).generate_from_frequencies(weights.to_dict())
        data = png_bytes(wc.to_image())
        st.image(data, caption="글자의 크기는 선택한 빈도 또는 TF·IDF를 나타냅니다.")
        st.download_button("워드클라우드 PNG 내려받기",data,"워드클라우드.png","image/png")
    except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.warning("그림을 만들 공간이 부족하거나 마스크를 읽지 못했습니다. 최소 글자 크기를 줄이거나 어두운 영역이 넓은 PNG를 사용해 주세요.")


if __name__ == "__main__":
    main()

