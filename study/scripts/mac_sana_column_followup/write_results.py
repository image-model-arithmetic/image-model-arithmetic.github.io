"""Write a readable result from completed, independently verified artifacts."""
import argparse,json
from pathlib import Path
from collect_completed import digest

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
final=root/'final-artifacts';report=json.loads((final/'final-report.json').read_text())
assert report['complete'] and report['integrity_passed']
assert report['independent_audit_sha256']==digest(root/'completed-evidence/independent-audit.json')
assert report['export_manifest_sha256']==digest(final/'export/manifest.json')
protocol=json.loads((final/'export/source/mac_sana_pixel_v1.json').read_text())
assert digest(final/'export/source/mac_sana_pixel_v1.json')==report['checkpoint']['pixel_protocol_sha256']
radix=2 if 'binary' in protocol else 10
outcome='Passed every registered independent mathematical gate.' if report['audited_success'] else 'Failed the registered independent mathematical gates.'
lines=[f'# {"Binary" if radix==2 else "Decimal"} addition with pretrained SANA','',f'**{outcome}**',
       '','All artifact-integrity checks passed. The table counts exact answers; consensus requires both the fixed template reader and independent Tesseract to be correct.',
       '','| Evaluation | Template exact | Reader consensus exact | Required |','|---|---:|---:|---:|']
for style,threshold in [('familiar','95%'),('unfamiliar','80%')]:
    primary=report['local'][style];independent=report['independent_local'][style]
    count=primary['count'];correct=primary['correct'];both=independent['consensus_correct']
    lines.append(f'| {style.capitalize()} input style | {correct}/{count} ({100*correct/count:.1f}%) | {both}/{count} ({100*both/count:.1f}%) | {threshold} |')
for length in sorted(report['programs'],key=int):
    primary=report['programs'][length];independent=report['independent_programs'][length]
    count=primary['count'];correct=primary['correct'];both=independent['consensus_correct']
    lines.append(f'| {length}-{"bit" if radix==2 else "digit"} programs | {correct}/{count} ({100*correct/count:.1f}%) | {both}/{count} ({100*both/count:.1f}%) | 90% |')
lines+=['','A program is correct only when every generated subtotal, all required image-validity checks, and the final sum are correct. Carry travels as the raw generated image crop. The model receives no OCR readout or symbolic answer during generation.',
        '',protocol['claim_boundary'],'',
        f"Selected update: **{report['checkpoint']['step']}**, with {report['checkpoint']['trainable_parameters']:,} trainable parameters in {report['checkpoint']['tensor_count']} tensors.",
        f"Full checkpoint SHA-256: `{report['checkpoint']['full_checkpoint_sha256']}`.",
        '','The exported inference graph uses the public pretrained SANA and its original VAE. The output-digit critic is a training-only loss and is absent from inference.',
        '','| Development control | Template addition | Consensus addition | Consensus copying |',
        '|---|---:|---:|---:|']
for name,metrics in report['baselines'].items():
    addition,copy=metrics['add'],metrics['copy']
    lines.append(f"| {name} | {addition['template_correct']}/{addition['count']} | {addition['consensus_correct']}/{addition['count']} | {copy['consensus_correct']}/{copy['count']} |")
demo=report['fixed_demo'];label='13-plus-7' if radix==2 else '27-plus-15'
lines+=['',f"The prospectively fixed exported-model example {'passed' if demo['correct'] else 'failed'} its arithmetic check; its artifact-integrity check {'passed' if demo['integrity_passed'] else 'failed'}.",
        f'[Generated answer image](final-artifacts/demo-{label}/answer.png) · [Full demo audit](final-artifacts/demo-audit.json)',
        '',f"The example's expected numerical sum is {demo['expected_sum']}; it is computed only by the external auditor.",
        '', '[Complete machine-readable report](final-artifacts/final-report.json) · [Independent per-item audit](completed-evidence/independent-audit.json) · [Exported inference instructions](final-artifacts/export/post-source/INFERENCE.md)',
        '',f"Export manifest SHA-256: `{report['export_manifest_sha256']}`.",
        f"Independent audit SHA-256: `{report['independent_audit_sha256']}`.",'']
target=root/'RESULTS.md';target.write_text('\n'.join(lines));print(str(target))
