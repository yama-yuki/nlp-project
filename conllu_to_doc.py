import spacy
from spacy_conll import init_parser
from spacy.tokens import Doc

def read_conllu(path):
    conll_list= []
    tmp = []
    with open(path, mode='r', encoding='utf-8') as d:
        lines = d.readlines()
        for line in lines:
            if line[0] != '#' and line[0] != '\n':
                tmp.append(line.rstrip('\n').split())
            else:
                if tmp:
                    conll_list.append(tmp)
                    tmp = []
        if tmp:
            conll_list.append(tmp)
    return conll_list

def conll_list_to_doc(vocab, conll):
    ## conll_list for a single sentence
    ## [['12', '31', '31', 'NUM', 'CD', 'NumType=Card', '13', 'nummod', '_', 'start_char=242|end_char=244'], 
    ## ['13', 'October', 'October', 'PROPN', 'NNP', 'Number=Sing', '10', 'obl', '_', 'start_char=245|end_char=252']]

    words, spaces, tags, poses, morphs, lemmas = [], [], [], [], [], []
    heads, deps = [], []
    for i in range(len(conll)):
        line = conll[i]
        parts = line
        id_, word, lemma, pos, tag, morph, head, dep, _, misc = parts

        if "." in id_ or "-" in id_:
            continue
        if "SpaceAfter=No" in misc:
            spaces.append(False)
        else:
            spaces.append(True)

        id_ = int(id_) - 1
        head = (int(head) - 1) if head not in ("0", "_") else id_
        tag = pos if tag == "_" else tag
        morph = morph if morph != "_" else ""
        dep = "ROOT" if dep == "root" else dep

        words.append(word)
        lemmas.append(lemma)
        poses.append(pos)
        tags.append(tag)
        morphs.append(morph)
        heads.append(head)
        deps.append(dep)

    doc = Doc(vocab, words=words, spaces=spaces)
    for i in range(len(doc)):
        doc[i].tag_ = tags[i]
        doc[i].pos_ = poses[i]
        doc[i].dep_ = deps[i]
        doc[i].lemma_ = lemmas[i]
        doc[i].head = doc[heads[i]]
    doc.is_parsed = True
    doc.is_tagged = True

    return doc

if __name__ == '__main__':
    nlp = init_parser("stanza",
                "en",
                parser_opts={"use_gpu": True, "verbose": False},
                include_headers=False)

    path = '<path to .conllu file>'
    conll = read_conllu(path)
    doc = conll_list_to_doc(nlp.vocab, conll)