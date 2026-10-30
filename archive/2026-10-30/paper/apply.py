import sys, re
sys.path.insert(0, '.')
from edits import E
src = open('/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/paper/paper_draft.tex').read()
# Table 1 range
a = src.index('\\begin{table}[tbp]'); b = src.index('\\label{tab:family}\n\\end{table}') + len('\\label{tab:family}\n\\end{table}')
assert src.count('\\begin{table}[tbp]') == 1
table1 = src[a:b]
bad = 0
for x in E:
    f = table1 if x['find'] == '@@TABLE1@@' else x['find']
    x['_find'] = f
    n = src.count(f)
    if n != 1:
        bad += 1; print('COUNT', n, '::', x['where'])
if bad: sys.exit('fix finds')
out = src
for x in E:
    assert out.count(x['_find']) == 1, x['where']
    out = out.replace(x['_find'], x['repl'])
open('/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/paper/paper_draft_revised.tex', 'w').write(out)
print('applied', len(E), 'edits')
# residual old numbers
for tok in ['0.0083','0.0274','0.0491','0.0158','0.0185','0.0123','0.0165','-0.65','-0.19','228 ','6.7M','nonzer ','TAs ','Future works','{T}','fig:lasso_coef','fig:shap_locus}','do not hold out','9/11','10/11 drugs']:
    for m in re.finditer(re.escape(tok), out):
        ln = out[:m.start()].count('\n') + 1
        line = out.split('\n')[ln-1]
        if line.lstrip().startswith('%'): continue
        print('residual', repr(tok), 'L', ln, line[max(0, m.start()-out.rfind('\n',0,m.start())-60):][:140])
