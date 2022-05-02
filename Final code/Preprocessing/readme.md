
* preprocess 
 
step 1: get text label list, pos:CWE-023...CWE-606; neg:NonVulnerable
step 2: create output path, full/raw/adj/label list/project name
step 3: generate mapping of text label and onehot label
step 4: create or load record.json(for incremental preprocessing)
step 5: handle data

* handle data
 
step1: for each dot file, skip already preprocessed dot file(in pre), get graphs from the rest
step2: for each subgraph in graphs, skip unvalid graph(node<2)
step3: write data to adj/nvocabf/graphlabel
step4: calculate removed graph number and valid graph number, print 
tep5: record total valid graph number to record.json
