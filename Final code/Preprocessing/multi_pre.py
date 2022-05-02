import pydot
import re
import os
# import numpy.random as random
import json
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing

# preprocess 
# step 1: get text label list, pos:CWE-023...CWE-606; neg:NonVulnerable
# step 2: create output path, full/raw/adj/label list/project name
# step 3: generate mapping of text label and onehot label
# step 4: create or load record.json(for incremental preprocessing)
# step 5: handle data

# handle data
# step1: for each dot file, skip already preprocessed dot file(in pre), get graphs from the rest
# step2: for each subgraph in graphs, skip unvalid graph(node<2)
# step3: write data to adj/nvocabf/graphlabel
# step4: calculate removed graph number and valid graph number, print 
# step5: record total valid graph number to record.json

def create_output_folder(out_path, subfolder_name, text_label):
    for label in text_label:        
        path = out_path + '/' + subfolder_name + '/' + label[0] 
        if not os.path.exists(path):
            os.makedirs(path)
            

def preprocess(folder_path, model):
    print('------------------------------------')
    print('start preprocessing')
    
    print('get text label list')
    #text_label: [[text_label1],[text_label2]...] 
    text_label = []
    for root, dirs, files in os.walk(folder_path + "/data"):
        for file in sorted(files):
            if file.endswith('dot'):
                if 'CWE' in file:
                    text_label.append([file[:-4]])
    #negative only have one label as NonVulnerable
    text_label.append(['NonVulnerable'])
    
    print('create output path')  
    out_path = folder_path + "/full/raw"
    create_output_folder(out_path, "adj", text_label)
    create_output_folder(out_path, "nvocabf", text_label)
    create_output_folder(out_path, "graphlabel", text_label)
    
    print('generate mapping of text label and onehot label')
    enc = preprocessing.OneHotEncoder(sparse=False)
    #onehot_label: [1.0 0.0 0.0 ...]
    onehot_label = enc.fit_transform(text_label)  
    mapping = {}
    for i in range(len(text_label)):
        mapping[text_label[i][0]]=''.join(str(list(map(int,onehot_label[i])))[1:-1].split(' '))
    #mapping: {text_label:'1,0,0'}
    print(mapping)
    
    print('create or load record.json')
    #record.json: {'text_label':node_num,...}
    #if record.json not exist, create an empty one
    if not os.path.exists(os.path.join(out_path, 'record.json')):
        f = open(os.path.join(out_path, 'record.json'), 'w+')
        f.close()
    #if record.json not empty, load last prerprocess data to pre
    with open(os.path.join(out_path, 'record.json'), 'r+') as f:
        if f.read() != '':
            f.seek(0)
            pre = json.load(f)
        else:
            pre = {}    
    pre_data = sum(pre.values())
    
    sbert_model = SentenceTransformer(model)
    #sbert_model = None
    
    print('handle data')
    removed0, removed1, unremoved, = 0, 0, 0
    removed_total, unremoved_total = 0, pre_data
    for root, dirs, files in os.walk(folder_path + "/data"):
        for file in sorted(files):
            #skip data already in pre
            if not file.endswith(".dot") or file in pre:
                continue
        
            removed0, removed1, unremoved, = 0, 0, 0
            file_path = os.path.join(root, file)
            file_name = file if 'NonVulnerable' not in file else 'NonVulnerable.dot'
            graphs = pydot.graph_from_dot_file(file_path)
            for graph in graphs:
                subgraphs = graph.get_subgraphs()
                for subgraph in subgraphs:
                    if len(subgraph.get_nodes()) < 2:
                        if len(subgraph.get_nodes()) == 0:
                            removed0 += 1
                        else:
                            removed1 += 1
                        continue
                    #filename: filename+valid data in this dot file+total valid data+subgraph name
                    newFileName = file_name[:-4] + '_' + str(unremoved + 1).rjust(5, "0") + '_' + str(
                                 unremoved_total + unremoved + 1).rjust(5, "0") + '_' + subgraph.get_name()
                    edges = subgraph.get_edges()
                    nodes = subgraph.get_nodes()
                    edgesArray = []
                    for edge in edges:
                        source = int(str(edge.get_source().strip('\"')).split('.')[1]) + 1
                        dest = int(str(edge.get_destination().strip('\"')).split('.')[1]) + 1
                        edgesArray.append(str(source) + "," + str(dest))
                    with open(out_path + '/adj/'  + file_name[:-4] + '/' + newFileName + '.adj', 'w') as f:
                        for i in edgesArray:
                            f.write(f"{i}" + "\n")
                    querys = []
                    # '\s<[0-9][0-9]*>[0-9][0-9]*$'
                    for node in nodes:
                        if not node.get_attributes():
                            continue
                        i = node.get_attributes()['label']
                        j = re.split('\s<[0-9][0-9]*>$', i)[0]
                        sentence_embeddings = sbert_model.encode(j, show_progress_bar=False)
                        #sentence_embeddings = [1,2,3]
                        c = ""
                        for d in sentence_embeddings:
                            c = c + str(d) + ","
                        querys.append(c[0:-1])
                    with open(out_path + '/nvocabf/'  + file_name[:-4] + '/' + newFileName  + '.ndf', 'w') as f:
                        for e in querys:
                            f.write(f"{e}\n")

                    with open(out_path + '/graphlabel/'  + file_name[:-4] + '/' + newFileName + '.gf', 'w') as f:
                        f.write(mapping[file_name[:-4]]+'\n')
                    
                    unremoved += 1
                    
            print('filename: ' + file_name + '\tremoved graphs: ' + str(removed0 + removed1) + '\tone node graphs:' + str(
                removed1) + '\tvalid graphs: ' + str(unremoved))
            
            # record
            with open(os.path.join(out_path, 'record.json'), 'r+') as f:
                if f.read() != '':
                    f.seek(0)
                    record = json.load(f)
                else:
                    record = {}
                record[file_name] = unremoved
                # clear record, dump new record
                f.seek(0)
                f.truncate()
                json.dump(record, f)            
            
        removed_total += removed0 + removed1
        unremoved_total += unremoved
    
    print('total graphs: ' + str(removed_total + unremoved_total))
    print('total removed graphs: ' + str(removed_total))
    print('total valid graphs: ' + str(unremoved_total))


if __name__ == '__main__':
    #preprocess('/Users/zhouqixin/Downloads/', "bert-base-nli-mean-tokens")
    preprocess("E:\\project\\multi_final", "bert-base-nli-mean-tokens")
    #preprocess("C:\\Users\\ruoya\\Desktop\\tt", "bert-base-nli-mean-tokens")

