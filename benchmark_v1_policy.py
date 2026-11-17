"""Frozen metadata mappings for the local FiveTrace benchmark, not model code."""
import re
import unicodedata

VERSION='fivetrace-standard-v1-20261115'
SEED=20261115


def canonical(text):
    return re.sub(r'[\W_]+','',unicodedata.normalize('NFKC',text).casefold())


def language(text):
    letters=[c for c in text if c.isalnum()]
    share=sum('\u3400'<=c<='\u9fff' for c in letters)/max(1,len(letters))
    return 'zh' if share>=.35 else 'en' if share<.01 else 'mixed'


def generator(value,source=''):
    raw=str(value or 'unknown');v=raw.lower().replace('_','-').replace(' ','').removeprefix('openai-')
    if v=='human':return 'human','human','declared'
    aliases={'gpt4o':'gpt-4o','gpt4':'gpt-4','gpt35':'gpt-3.5','gpt-35':'gpt-3.5',
             'chatgpt':'chatgpt-unversioned','chatgpt-early':'chatgpt-early','gpt4.1':'gpt-4.1',
             'llma':'llama-unversioned','claude':'claude-unversioned','qwen':'qwen-unversioned',
             'llama':'llama-unversioned','gemini':'gemini-unversioned'}
    v=aliases.get(v,v).removeprefix('openai-')
    if any(x in v for x in ['llada','diffusiongemma','dream','sdar']):
        fam='llada' if 'llada' in v else 'diffusiongemma' if 'diffusiongemma' in v else 'dream' if 'dream' in v else 'sdar'
    elif 'gpt-j' in v or 'gpt-neox' in v:fam='eleutherai'
    elif 'claude' in v:fam='anthropic'
    elif 'qwen' in v:fam='qwen'
    elif 'deepseek' in v:fam='deepseek'
    elif 'gemini' in v:fam='gemini'
    elif 'gemma' in v:fam='gemma'
    elif 'doubao' in v:fam='doubao'
    elif any(x in v for x in ['gpt-','chatgpt','davinci']) or re.match(r'^o[134](?:-|$)',v):fam='openai'
    elif 'llama' in v or 'vicuna' in v:fam='llama'
    elif 'glm' in v:fam='glm'
    elif 'baichuan' in v:fam='baichuan'
    elif 'bloom' in v:fam='bloom'
    elif 'flan' in v:fam='flan_t5'
    elif 'opt-' in v:fam='opt'
    elif 'dolly' in v:fam='dolly'
    elif 'cohere' in v:fam='cohere'
    elif 'mixtral' in v or 'mistral' in v:fam='mistral'
    elif 'stablelm' in v:fam='stablelm'
    else:fam='unknown'
    confidence='unknown' if fam=='unknown' else 'family_only' if 'unversioned' in v or v in ['qwen-max','gpt-4','gpt-3.5'] else 'source_reported'
    return v,fam,confidence


def domain(value):
    v=str(value or '').lower().replace(' ','_')
    if any(x in v for x in ['paper','academic','arxiv','pubmed','peerread','sci_gen','csl','review_of_research','peer_review']):return 'academic'
    if any(x in v for x in ['news','cnn','xsum','cnewsum']):return 'news'
    if any(x in v for x in ['novel','story','fiction','storal','creative','roct','hswag','writing_prompt','mage/wp']):return 'narrative'
    if any(x in v for x in ['film','review','imdb','yelp','cmv','opinion','composition','essay','argument']):return 'opinion'
    if any(x in v for x in ['wiki','seo','webtext','explan','encyclopedia','instruction']):return 'expository'
    if any(x in v for x in ['qa','question','answer','eli5','squad','finance','medicine','psychology','law','baike','reddit','open_qa','computer','nlpcc']):return 'qa'
    return 'other'


def length_bin(language,characters):
    # Language-specific character bins; no model-dependent tokenizer in a data split.
    edges=(300,800) if language=='zh' else (900,2400)
    return 'short' if characters<edges[0] else 'medium' if characters<edges[1] else 'long'
