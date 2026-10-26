"""Prospective paired raw-carry interventions, with no model-dependent case selection."""
from collections import Counter


def cases(config):
    assert config['donor_pairs'] == [ [[0,0,0],[1,1,0]], [[0,1,0],[1,1,1]],
                                    [[1,0,0],[0,1,1]], [[0,0,1],[1,0,1]] ]
    donors=[]
    for bank, pair in enumerate(config['donor_pairs']):
        for carry, (a,b,c) in enumerate(pair):
            assert (a+b+c)//2 == carry
            donors.append({'id':f'donor-{bank}-{carry}','bank':bank,'carry':carry,
                           'a':a,'b':b,'c':c,'seed':config['donor_seed']+2*bank+carry})
    pairs=[{'id':f'pair-{bank}-{a}{b}-{repeat:02d}','bank':bank,'a':a,'b':b,'repeat':repeat,
            'seed':config['receiver_seed']+bank*1000+a*100+b*50+repeat,
            'zero_donor':f'donor-{bank}-0','one_donor':f'donor-{bank}-1','repeat_zero':repeat==0}
           for bank in range(4) for a in range(2) for b in range(2) for repeat in range(5)]
    return {'donors':donors,'pairs':pairs}


def audit_corpus(config):
    corpus=cases(config);donors,pairs=corpus['donors'],corpus['pairs']
    assert len(donors)==8 and len(pairs)==80
    assert len({r['id'] for r in donors+pairs})==88
    assert len({r['seed'] for r in donors+pairs})==88
    assert {(r['a'],r['b'],r['c']) for r in donors} == {
        (a,b,c) for a in range(2) for b in range(2) for c in range(2)}
    assert Counter((r['a'],r['b']) for r in pairs)=={(a,b):20 for a in range(2) for b in range(2)}
    assert Counter(r['bank'] for r in pairs)=={i:20 for i in range(4)}
    repeats=sum(r['repeat_zero'] for r in pairs)
    assert repeats==16
    return {'passed':True,'donors':8,'counterfactual_pairs':80,'repeat_controls':repeats,
            'images':len(donors)+2*len(pairs)+repeats,'unique_noise_seeds':88,
            'ignore_carry_maximum_correct_pairs':0,'scope':'Prospective corpus and shortcut audit only'}
