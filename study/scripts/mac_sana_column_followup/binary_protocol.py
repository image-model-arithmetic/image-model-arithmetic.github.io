"""Prospective binary corpus and post-generation score definitions."""
from collections import Counter
import numpy as np


def display_label(total):
    assert 0 <= total <= 3
    return 10*(total//2)+total%2


def training_rows():
    return [{'id':f'{"copy" if copy else "add"}-{a}-{b}-{c}-v{variant}',
             'a':a,'b':b,'c':c,'variant':variant,'copy':copy,
             'answer':a if copy else display_label(a+b+c)}
            for copy in [True,False] for variant in range(4) for a in range(2)
            for b,c in ([(0,0)] if copy else [(b,c) for b in range(2) for c in range(2)])]


def programs(seed):
    rng=np.random.default_rng(seed)
    rows=[]
    for length in [4,8,16]:
        seen=set()
        while len(seen)<32:
            aa=[int(rng.integers(0,2)) for _ in range(length)]
            bb=[int(rng.integers(0,2)) for _ in range(length)]
            aa[-1]=bb[-1]=1
            left=sum(v*2**i for i,v in enumerate(aa))
            right=sum(v*2**i for i,v in enumerate(bb))
            if (left,right) in seen:continue
            seen.add((left,right))
            rows.append({'id':f'program-{length}-{len(seen)-1:03d}',
                         'length':length,'left':left,'right':right,'a_bits':aa,'b_bits':bb})
    return rows


def binary_read(im,banks,read_answer):
    readings=[read_answer(im,bank) for bank in banks]
    result=min(readings,key=lambda r:r['carry']['mse']+r['digit']['mse'])
    result['valid']=result['valid'] and result['digit']['prediction']<=1
    result['numeric_prediction']=2*result['carry']['prediction']+result['digit']['prediction']
    return result


def audit_corpus(seed):
    rows=training_rows();facts={(r['a'],r['b'],r['c']) for r in rows if not r['copy']}
    assert len(rows)==40 and len({r['id'] for r in rows})==40 and len(facts)==8
    labels=Counter(display_label(sum(fact)) for fact in facts)
    corpus=programs(seed);by_length={};failures=[]
    for length in [4,8,16]:
        subset=[p for p in corpus if p['length']==length]
        assert len(subset)==32 and len({(p['left'],p['right']) for p in subset})==32
        ignore=copy_left=copy_right=0;carry_stages=0
        for p in subset:
            assert p['left'].bit_length()==p['right'].bit_length()==length
            carry=0;without=0
            for i,(a,b) in enumerate(zip(p['a_bits'],p['b_bits'])):
                carry_stages+=carry;carry=(a+b+carry)//2
                without|=((a+b)%2)<<i
            without|=((p['a_bits'][-1]+p['b_bits'][-1])//2)<<length
            ignore+=without==p['left']+p['right']
            copy_left+=p['left']==p['left']+p['right']
            copy_right+=p['right']==p['left']+p['right']
        by_length[str(length)]={'count':32,'unique_operand_pairs':32,'stages':32*length,
            'carry_one_stages':carry_stages,'ignore_carry_oracle_correct':ignore,
            'copy_left_correct':copy_left,'copy_right_correct':copy_right}
    return {'passed':not failures,'training_rows':40,'elementary_facts':8,
            'local_label_counts':dict(sorted(labels.items())),
            'best_constant_local_correct_out_of_8':max(labels.values()),
            'ignore_carry_local_correct_out_of_8':sum(display_label(a+b)==display_label(a+b+c) for a,b,c in facts),
            'program_seed':seed,'program_count':96,'by_length':by_length,'failures':failures,
            'scope':'Prospective label/shortcut audit only; eight supervised binary facts, with fresh-noise/style and new-composition evaluation'}
