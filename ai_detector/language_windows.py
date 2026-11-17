"""Language-aware outer windows; source offsets and paragraph counts stay intact."""
from bisect import bisect_right
import re


def english_token_windows(text,tokenizer,max_tokens=384,min_tail_tokens=128):
    if not 1<=min_tail_tokens<=max_tokens:raise ValueError('Invalid token window/tail limits')
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,truncation=False)
    offsets=encoded['offset_mapping'];n=len(offsets)
    if not n:return [(0,len(text))]
    ends=[int(end) for start,end in offsets]
    boundaries=[m.end() for m in re.finditer(r'[.!?][\"\'\)\]]*\s+|\n+',text)]
    spans=[];start_token=0;start_char=0
    while start_token<n:
        stop=min(n,start_token+max_tokens)
        # A short final token slice is merged, keeping the result <=max+tail-1.
        if 0<n-stop<min_tail_tokens:stop=n
        end_char=len(text) if stop==n else ends[stop-1]
        if stop<n:
            lower=ends[min(n-1,start_token+max_tokens//2)]
            choices=[b for b in boundaries if lower<=b<=end_char]
            if choices:
                candidate=max(choices);candidate_stop=bisect_right(ends,candidate)
                if candidate_stop>start_token:
                    stop=candidate_stop;end_char=candidate
                    if 0<n-stop<min_tail_tokens:stop=n;end_char=len(text)
        if end_char<=start_char:raise ValueError('Tokenizer offsets do not advance')
        spans.append((start_char,end_char));start_char=end_char;start_token=stop
    return spans


def language_window_spans(text,tokenizer,chinese_characters=600,max_tokens=384,min_tail_tokens=128):
    from .features import english_route
    from .paragraphs import document_windows
    if english_route(text):return english_token_windows(text,tokenizer,max_tokens,min_tail_tokens)
    return document_windows(text,max_chars=chinese_characters)


def language_document_structure(raw_text,tokenizer):
    from .paragraphs import prepare_detection_paragraphs,detection_window_spans
    from .data import normalize
    from .expanded_data import clean_text
    import hashlib
    normalized,sources,paragraphs=prepare_detection_paragraphs(raw_text)
    result=[]
    for p in paragraphs:
        windows=[]
        splitter=lambda t:language_window_spans(t,tokenizer)
        for start,end in detection_window_spans(p['text'],window_chars=600,window_splitter=splitter):
            text=p['text'][start:end]
            key=hashlib.sha256(normalize(clean_text(text)).encode('utf-8')).hexdigest()
            windows.append(dict(key=key,text=text,characters=sum(not c.isspace() for c in text)))
        result.append(dict(paragraph=p['paragraph'],characters=p['characters'],windows=windows))
    return result
