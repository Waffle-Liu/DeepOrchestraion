# DeepOrchestration

> It is a cool project using deep learning to help ochestration. Fighting!

### Generate the dataset
- download the raw data from http://www.orch-idea.org/ 
- select all ord files played by ['Vc', 'Fl', 'Va', 'Vn', 'Ob', 'BTb',
       'Cb', 'ClBb', 'Hn', 'TpC', 'Bn', 'Tbn'] and put these files in a shared directory
- call method `show_all_class_num` in `process_Stat.py` to get the `class.index`
- set paras in `process_Stat.py` 
- call method `random_combine` to generate new datasets

### Train a network
Just run `python main.py`. Before training, make sure that the dictionary `db` is right. If it is a new dataset, call method `stat_test_db` in `process_Stat.py` to get the new `db`.

