import json

def load_maia(
    path: str,
):
    train = []
    val = []
    test = []
    
    for (file_f, file_d) in [("pt_br_client_04_a.json", "dBR2_splits_a.json"),("pt_br_client_02_a.json", "dBR1_splits_a.json"),("pt_pt_client_03_a.json", "dPT_splits_a.json"),("de_client_01_a.json", "dDE1_splits_a.json"),("de_client_02_a.json", "dDE2_splits_a.json")]:
    
        f = open("data/maia/"+file_f)
        d = open("data/maia/"+file_d)
        pointers = json.load(d)
        dataset = json.load(f)

        pointers_train = []
        pointers_val = []
        pointers_test = []
        
        for i in range(len(dataset)):
            dial_id = dataset[i]["id"]
            
            if dial_id in pointers["test"]:
                pointers_test.append(i)
            elif dial_id in pointers["val"]:
                pointers_val.append(i)
            else:
                pointers_train.append(i)


        for d_id in pointers_train:
            for id, turn in enumerate(dataset[d_id]["turns"]): 
                for i in range (len(turn['text_src'])):
                    label = turn['Emotion'][i]
                    if label >7 or label <0:
                        continue
                    train.append({'text': turn['text_src'][i], "label": label, "dialog_id": d_id})
            
        for d_id in pointers_val:
            for id, turn in enumerate(dataset[d_id]["turns"]): 
                for i in range (len(turn['text_src'])):
                    label = turn['Emotion'][i]
                    if label >7 or label <0:
                        continue
                    val.append({'text': turn['text_src'][i], "label": label, "dialog_id": d_id})
                    
        for d_id in pointers_test:
            for id, turn in enumerate(dataset[d_id]["turns"]): 
                for i in range (len(turn['text_src'])):
                    label = turn['Emotion'][i]
                    if label >7 or label <0:
                        continue
                    test.append({'text': turn['text_src'][i], "label": label, "dialog_id": d_id})
    
    return (train, val, test) 

