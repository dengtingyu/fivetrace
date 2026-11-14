"""Paragraph counting and report generation, independent of the model runtime."""
import math

LABELS = {"ai": "偏向 AI 生成", "human": "偏向人工撰写", "uncertain": "不确定"}
BINARY_LABELS = {"ai": "AI", "human": "人工"}
REASONS = {"classified": "依据第 1–5 项综合判断",
           "merged_short": "短段与相邻段合并后检测",
           "combined_windows": "长段内部片段分数汇总后判定",
           "merged_short_and_windows": "短段合并，内部片段分数汇总后判定"}


def document_windows(text, max_chars=1000, min_tail=100):
    spans, start = [], 0
    while start < len(text):
        end = min(len(text), start+max_chars)
        if end < len(text):
            boundary = max(text.rfind(mark, start+max_chars//2, end) for mark in "。！？\n")
            if boundary >= 0:
                end = boundary+1
        spans.append((start, end))
        start = end
    if len(spans) > 1 and spans[-1][1]-spans[-1][0] < min_tail:
        spans[-2:] = [(spans[-2][0], spans[-1][1])]
    return spans


def score_category(score, thresholds):
    if score >= thresholds["ai_at_or_above"]:
        return "ai"
    if score < thresholds["human_below"]:
        return "human"
    return "uncertain"


def verdict(score, thresholds):
    return LABELS[score_category(score, thresholds)]


def split_paragraphs(raw_text):
    """Each nonempty original text line is one paragraph; keep canonical offsets."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs, offset = [], 0
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        body = line.strip()
        if body:
            start = offset + len(line)-len(line.lstrip())
            end = offset + len(line.rstrip())
            paragraphs.append({"paragraph": len(paragraphs)+1, "line": line_number,
                               "start": start, "end": end, "text": text[start:end],
                               "characters": sum(not c.isspace() for c in body)})
        offset += len(line)
    return text, paragraphs


def _merge_short_units(units, min_chars):
    groups, pending, pending_chars = [], [], 0
    for unit in units:
        if not pending and unit["characters"] >= min_chars:
            groups.append([unit])
        else:
            pending.append(unit)
            pending_chars += unit["characters"]
            if pending_chars >= min_chars:
                groups.append(pending)
                pending, pending_chars = [], 0
    if pending:
        if not groups:
            raise ValueError(f"正文不足 {min_chars} 个非空白字符，请提供更多正文。")
        groups[-1].extend(pending)
    return groups


def prepare_detection_paragraphs(raw_text, min_chars=100):
    text, source = split_paragraphs(raw_text)
    if not source:
        raise ValueError("请提供正文；空行不计为段落。")
    paragraphs = []
    for index, group in enumerate(_merge_short_units(source, min_chars), start=1):
        start, end = group[0]["start"], group[-1]["end"]
        body = text[start:end]
        paragraphs.append({"paragraph": index, "source_paragraphs": [p["paragraph"] for p in group],
                           "source_lines": [p["line"] for p in group], "start": start, "end": end,
                           "text": body, "characters": sum(not c.isspace() for c in body)})
    return text, source, paragraphs


def summarize_paragraphs(paragraphs):
    total = len(paragraphs)
    if not total:
        raise ValueError("没有可统计的非空段落。")
    counts = {name: sum(p["category"] == name for p in paragraphs) for name in BINARY_LABELS}
    if sum(counts.values()) != total:
        raise ValueError("存在未知段落类别。")
    # Largest remainder allocation makes the displayed one-decimal percentages sum to 100%.
    tenths = {name: counts[name]*1000//total for name in BINARY_LABELS}
    order = sorted(BINARY_LABELS, key=lambda name: counts[name]*1000 % total, reverse=True)
    for name in order[:1000-sum(tenths.values())]:
        tenths[name] += 1
    return {"total_paragraphs": total, **{
        name: {"count": counts[name], "ratio": counts[name]/total, "percentage": tenths[name]/10}
        for name in BINARY_LABELS}}


def analyze_paragraphs(raw_text, predict_window, thresholds, reference_model, progress=None, min_chars=100, window_chars=1000):
    text, source_paragraphs, paragraphs = prepare_detection_paragraphs(raw_text, min_chars)
    results = []
    for source in paragraphs:
        if progress:
            progress(source["paragraph"], len(paragraphs))
        windows = []
        spans = [{"start": start, "end": end,
                  "characters": sum(not c.isspace() for c in source["text"][start:end])}
                 for start, end in document_windows(source["text"], max_chars=window_chars)]
        for index, group in enumerate(_merge_short_units(spans, min_chars), start=1):
            start, end = group[0]["start"], group[-1]["end"]
            part = source["text"][start:end]
            chars = sum(not c.isspace() for c in part)
            try:
                scored = predict_window(part)
                score = scored["ai_tendency_score"]
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("模型返回了无效分数")
            except Exception as error:
                raise RuntimeError(f"第 {source['paragraph']} 段的内部片段 {index} 检测失败：{error}") from error
            category = "ai" if score >= thresholds["ai_at_or_above"] else "human"
            windows.append({**scored, "window": index, "start": start, "end": end,
                            "text": part, "characters": chars, "category": category,
                            "verdict": BINARY_LABELS[category]})
        score = sum(w["ai_tendency_score"]*w["characters"] for w in windows)/sum(w["characters"] for w in windows)
        category = "ai" if score >= thresholds["ai_at_or_above"] else "human"
        merged = len(source["source_paragraphs"]) > 1
        reason = ("merged_short_and_windows" if merged else "combined_windows") if len(windows) > 1 else ("merged_short" if merged else "classified")
        results.append(dict(source, category=category, verdict=BINARY_LABELS[category], reason=reason,
                            ai_tendency_score=score,
                            text_classifier_score=windows[0]["text_classifier_score"] if len(windows) == 1 else None,
                            features=windows[0]["features"] if len(windows) == 1 else None, windows=windows))
    summary = summarize_paragraphs(results)
    return {"report_version": 2, "model": "Chinese five-feature paragraph detector",
            "feature_groups_used": [1, 2, 3, 4, 5], "reference_model": reference_model,
            "classification_mode": "binary", "characters": len(text),
            "source_paragraphs": len(source_paragraphs), "total_paragraphs": len(results), "paragraphs": results,
            "summary": summary, "ai_paragraph_ratio": summary["ai"]["ratio"],
            "human_paragraph_ratio": summary["human"]["ratio"],
            "statistics_basis": "detection_paragraph_count_after_short_merge", "paragraph_definition": "nonempty_original_lines_short_merged",
            "minimum_scored_characters": min_chars, "thresholds": thresholds,
            "internal_window_characters": window_chars,
            "notes": ["只输出人工／AI；占比按短段合并后的检测段数量统计，两项合计 100%。",
                      "每个原文非空行视作一个自然段；不足 100 字的短段并入相邻段，空行跳过。",
                      "长段内部片段不增加段落总数，内部片段分数汇总后给出整段二分类。",
                      "每段分数用于查看证据；文章的 AI 率定义为疑似 AI 段落数占比。",
                      "段落模型主要用论文摘要、引言训练，整篇文章的占比是本次分类统计。"]}


def report_to_markdown(report):
    summary = report["summary"]
    lines = ["# 整篇文章段落检测报告", "", f"共 {summary['total_paragraphs']} 个检测段；每段使用第 1–5 项综合判断。", "",
             "| 类别 | 段落数 | 段落占比 |", "|---|---:|---:|"]
    for key, name in [("ai", "AI 段落"), ("human", "人工段落")]:
        item = summary[key]
        lines.append(f"| {name} | {item['count']} | {item['percentage']:.1f}% |")
    lines += ["", f"原文有 {report['source_paragraphs']} 个非空自然段，短段合并后有 {report['total_paragraphs']} 个检测段。",
              "统计口径：空行不计；短段并入相邻段；长段内部切片不单独计数。两项占比按检测段数统计。", "",
              "## 逐段结果", "", "| 段落 | 判断 | 非空白字符 | 参考分数 | 原因 |", "|---|---|---:|---:|---|"]
    for paragraph in report["paragraphs"]:
        score = paragraph["ai_tendency_score"]
        rendered_score = f"{score:.4f}" if score is not None else "—"
        lines.append(f"| {paragraph['paragraph']} | {paragraph['verdict']} | {paragraph['characters']} | {rendered_score} | {REASONS[paragraph['reason']]} |")
    lines += ["", "## 段落明细", ""]
    for paragraph in report["paragraphs"]:
        lines += [f"### 第 {paragraph['paragraph']} 段 · {paragraph['verdict']}", "", REASONS[paragraph["reason"]], "",
                  "> " + paragraph["text"].replace("\n", "\n> "), ""]
        if len(paragraph["windows"]) > 1:
            lines += ["内部片段：", ""]
            for window in paragraph["windows"]:
                score = window["ai_tendency_score"]
                rendered_score = f"{score:.4f}" if score is not None else "—"
                lines.append(f"- {window['window']}：{window['verdict']}，分数 {rendered_score}")
            lines.append("")
    lines += ["## 说明", ""] + [f"- {note}" for note in report["notes"]]
    return "\n".join(lines) + "\n"
