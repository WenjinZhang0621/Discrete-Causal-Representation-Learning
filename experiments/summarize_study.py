import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dcrl.study_summary import summarize_study

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    args=p.parse_args()
    result=summarize_study(args.directory)
    print(f"Summarized {result['completed']}/{result['planned']} datasets; {len(result['failures'])} failed/partial.")
