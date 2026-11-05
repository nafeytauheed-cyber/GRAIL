import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='GRAIL traffic-disruption research')
    s=p.add_subparsers(dest='command',required=True)
    prep=s.add_parser('prepare');prep.add_argument('csv');prep.add_argument('parquet')
    au=s.add_parser('audit');au.add_argument('parquet');au.add_argument('--out',default='reports/audit')
    ex=s.add_parser('experiment');ex.add_argument('--parquet');ex.add_argument('--synthetic',action='store_true');ex.add_argument('--rows',type=int,default=8000)
    ex.add_argument('--out',default='reports/results/synthetic');ex.add_argument('--seeds',default='11,23,37');ex.add_argument('--modes',default='temporal,geographic');ex.add_argument('--models')
    ex.add_argument('--epochs',type=int,default=25);ex.add_argument('--trials',type=int,default=2);ex.add_argument('--feature-mode',choices=['primary','no_geo','provider','year','no_weather'],default='primary')
    ex.add_argument('--source');ex.add_argument('--start');ex.add_argument('--end');ex.add_argument('--label-review',help='Path to JSON recording dataset compatibility review; mandatory for real data')
    fi=s.add_parser('figures');fi.add_argument('results');fi.add_argument('--out',default='reports/figures')
    args=p.parse_args()
    if args.command=='prepare':
        from .data.warehouse import prepare
        prepare(args.csv,args.parquet)
    elif args.command=='audit':
        from .data.warehouse import audit
        print(json.dumps(audit(args.parquet,args.out),indent=2))
    elif args.command=='figures':
        from .training.experiment import figures
        figures(args.results,args.out)
    else:
        from .training.experiment import run
        if args.synthetic==bool(args.parquet):p.error('Choose exactly one of --synthetic or --parquet')
        if args.rows<100:p.error('At least 100 rows required')
        if args.synthetic:
            from .data.synthetic import synthetic
            df=synthetic(args.rows)
        else:
            if not args.label_review:p.error('Review full audit first; provide --label-review with a defensible scope and limitations')
            review=json.loads(Path(args.label_review).read_text())
            if not review.get('rationale') or review.get('status') not in ['comparable','exploratory_unverified']:
                p.error('Label review requires rationale and status comparable or exploratory_unverified')
            from .data.warehouse import load_development_sample
            df=load_development_sample(args.parquet,args.rows,source=args.source,start=args.start,end=args.end)
            Path(args.out).mkdir(parents=True,exist_ok=True);(Path(args.out)/'label_review.json').write_text(json.dumps(review,indent=2))
        run(df,args.out,seeds=[int(x) for x in args.seeds.split(',')],modes=args.modes.split(','),names=args.models.split(',') if args.models else None,
            epochs=args.epochs,trials=args.trials,synthetic=args.synthetic,feature_mode=args.feature_mode)
if __name__=='__main__':main()
