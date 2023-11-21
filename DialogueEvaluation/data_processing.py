import argparse
import csv
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Process data for dialogue evaluation')
    parser.add_argument('json', type=str, help='path to input json file')
    args = parser.parse_args()

    data = json.load(open(args.json))
    filename = args.json.split(".")[0]

    ctx_src = []
    res_src = []
    ctx_tgt = []
    res_tgt = []
    correctness = []
    templated = []
    engagement = []
    understanding = []
    sensibleness = []
    politeness = []
    iq = []
    dropped=[]
    sucess=[]
    turn_count=[]

    for dialog in data:
        ctx_src = []
        ctx_tgt = []

        dropped.append(dialog['dialog']['Dropped conversation'])
        sucess.append(dialog['dialog']['Task Sucess'])
        turn_count.append(len(dialog['turns'])-1)

        for i in range(1,len(dialog['turns'])):
            prev = dialog['turns'][i-1]
            cur = dialog['turns'][i]

            if cur["floor"] == 'inbound': 
                ctx_src.append('<speaker1>'+' '.join(prev['text_mt']))
                ctx_tgt.append('<speaker1>'+' '.join(prev['text_src']))

                res_src.append('<speaker2>'+' '.join(cur['text_src']))
                res_tgt.append('<speaker2>'+' '.join(cur['text_mt']))

            else:
                ctx_src.append('<speaker2>'+' '.join(prev['text_src']))
                ctx_tgt.append('<speaker2>'+' '.join(prev['text_mt']))

                res_src.append('<speaker2>'+' '.join(cur['text_mt']))
                res_tgt.append('<speaker2>'+' '.join(cur['text_src']))
            try:
                c = np.nanmin(cur['Correctness'])
            except Exception:
                c = None
            try:
                t = np.nanmin(cur['Templated'])
            except Exception:
                t = None
            try:
                e = np.nanmin(cur['Engagement'])
            except Exception:
                e = None

            correctness.append(c)
            templated.append(t)
            engagement.append(e)
            understanding.append(cur['Understanding'])
            sensibleness.append(cur['Sensibleness'])
            politeness.append(cur['Politeness'])
            iq.append(cur['IQ'])   

    qa_data_turn = {'ctx_src': ctx_src,
               'ctx_tgt': ctx_tgt,
               'res_src': res_src,
               'res_tgt': res_tgt,
               'correctness': correctness,
               'templated': templated,
               'engagement':engagement,
               'understanding':understanding,
               'sensibleness':sensibleness,
               'politeness':politeness,
               'iq':iq,
              }

    qa_data_dial = {'dropped': dropped,
                    'sucess': sucess,
                    'turn_count': turn_count
                }

    with open(f"{filename}_turn.csv", "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(qa_data_turn.keys())
        writer.writerows(zip(*qa_data_turn.values()))


    with open(f"{filename}_dial.csv", "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(qa_data_dial.keys())
        writer.writerows(zip(*qa_data_dial.values()))
        
if __name__ == '__main__':
    main()