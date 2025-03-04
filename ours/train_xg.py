import json
import numpy as np
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import openai
import random
import os
import time
import copy
import argparse
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from utils_xg import evaluate, gen_prompt, tree_to_code, get_cart, evaluate_init, add_column, load_model, use_api
import re, torch
import importlib
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from params import xgb_params_dict

parser = argparse.ArgumentParser(description = 'ours')
parser.add_argument('--data_name', default = 'phoneme', type = str)
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--num_steps', default = 50, type = int)
parser.add_argument('--model_name', default = 'kykim0/Llama-2-7b-ultrachat200k-2e', type = str)
parser.add_argument('--step', default = 1, type = int)
args = parser.parse_args()

name = args.data_name
seed = args.seed
steps = args.num_steps

# fix seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer = load_model(args.model_name, None)

xtrain = np.load(f'../data/{name}/seed{seed}/xtrain{args.step-1}.npy')
xval = np.load(f'../data/{name}/seed{seed}/xval{args.step-1}.npy')
xtest = np.load(f'../data/{name}/seed{seed}/xtest{args.step-1}.npy')
ytrain = np.load(f'../data/{name}/seed{seed}/ytrain.npy')
yval = np.load(f'../data/{name}/seed{seed}/yval.npy')
ytest = np.load(f'../data/{name}/seed{seed}/ytest.npy')

param = xgb_params_dict[f"{name}/seed{seed}"]

train_acc_list = []
score_list = []
test_acc_list = []
r_list = []
dt_list = []

        
_, best_val, best_test = evaluate_init(xtrain, ytrain, xval, yval, xtest, ytest, param)

print("Step 0 | Val: {:.2f} | Test: {:.2f}".format(best_val*100, best_test*100))

# Train initial predictor
best_val = 0
for i in range(1, 12):
    clf = XGBClassifier(max_depth = i, tree_method = 'hist', random_state = 0, seed = 0, device = 'cuda')
    clf.fit(xtrain, ytrain)
    xtrain_pred = clf.predict(xtrain)
    xval_pred = clf.predict(xval)
    xtest_pred = clf.predict(xtest)
    train_acc = accuracy_score(xtrain_pred, ytrain)*100
    val_acc = accuracy_score(xval_pred, yval)*100
    test_acc = accuracy_score(xtest_pred, ytest)*100
    if val_acc > best_val:
        best_train, best_val, best_test = train_acc, val_acc, test_acc
        best_clf = copy.deepcopy(clf)

importance = best_clf.feature_importances_

sorting_imp = np.argsort(-importance)
r0 = "x{:.0f} = [x{:.0f} * x{:.0f}]".format(len(xtrain[0]) + 1, sorting_imp[0] + 1, sorting_imp[1] + 1)

def rule_template(rule):
    variables = "x1"
    for i in range(1, len(xtrain[0])):
        variables += ", x{:.0f}".format(i+1)
    target_variable = "x{:.0f}".format(len(xtrain[0])+1)
    text = f'''
import numpy as np

def rule(data):
    [{variables}] = data
    {rule}
    return {target_variable}[0]
    '''
    return text

rule_text = rule_template(r0)
with open(f"../data/{name}/seed{seed}/rule_list.py", "w") as f:
    f.write(rule_text)

exec(rule_text, globals())

new_col = [(rule(xtrain[i])) for i in range(len(xtrain))]
new_col += [(rule(xval[i])) for i in range(len(xval))]
new_col += [(rule(xtest[i])) for i in range(len(xtest))]

train_acc, val_acc, test_acc = evaluate(new_col, xtrain, ytrain, xval, yval, xtest, ytest, param)

best_CART = get_cart(new_col, xtrain, ytrain, xval, yval, seed) # Train CART

dt0 = tree_to_code(best_CART, ['x{}'.format(i) for i in range(1, len(xtrain[0]) + 2)]) # Tree to Text

# append
r_list.append(r0)
score_list.append(val_acc)
test_acc_list.append(test_acc)
dt_list.append(dt0)
train_acc_list.append(train_acc)

pattern = r"x{}\s*=\s*\[.*?\]".format(len(xtrain[0]) + 1)

# Optimize start
for step in range(steps):
    prompt = gen_prompt(r_list, dt_list, score_list, len(xtrain[0])+1)
    all_status = 0
    while all_status == 0:
        answer_temp1 = use_api(prompt, model, tokenizer, 1.0)

        for num_iter in range(len(answer_temp1)):
            try:
                os.remove(f"{name}/__pycache__/rule_list.cpython-38.pyc")
            except:
                pass
            
            match = re.search(pattern, answer_temp1[num_iter], re.DOTALL)
            
            if match:
                try:
                    extracted_text = match.group()
                    rule_text = rule_template(extracted_text)
                    exec(rule_text, globals())
                    new_col = [(rule(xtrain[i])) for i in range(len(xtrain))]
                    new_col += [(rule(xval[i])) for i in range(len(xval))]
                    new_col += [(rule(xtest[i])) for i in range(len(xtest))]
                    train_acc, val_acc, test_acc = evaluate(new_col, xtrain, ytrain, xval, yval, xtest, ytest, param)
                    best_CART = get_cart(new_col, xtrain, ytrain, xval, yval, seed) # Train CART
                    dt = tree_to_code(best_CART, ['x{}'.format(i) for i in range(1, len(xtrain[0]) + 2)])
                    
                    if val_acc > np.max(np.array(score_list)):
                        np.save(f"../data/{name}/seed{seed}/new_col_step{args.step-1}.npy", new_col)
                        new_xtrain, new_xval, new_xtest = add_column(xtrain, xval, xtest, new_col)
                        np.save(f"../data/{name}/seed{seed}/xtrain{args.step}.npy", new_xtrain)
                        np.save(f"../data/{name}/seed{seed}/xval{args.step}.npy", new_xval)
                        np.save(f"../data/{name}/seed{seed}/xtest{args.step}.npy", new_xtest)
                    elif val_acc == np.max(np.array(score_list)):
                        idxes = np.where(np.array(score_list) == val_acc)[0]
                        if train_acc < np.min(np.array(train_acc_list)[idxes]):
                            np.save(f"../data/{name}/seed{seed}/new_col_step{args.step-1}.npy", new_col)  
                            new_xtrain, new_xval, new_xtest = add_column(xtrain, xval, xtest, new_col)
                            np.save(f"../data/{name}/seed{seed}/xtrain{args.step}.npy", new_xtrain)
                            np.save(f"../data/{name}/seed{seed}/xval{args.step}.npy", new_xval)
                            np.save(f"../data/{name}/seed{seed}/xtest{args.step}.npy", new_xtest)                  
                    
                    r_list.append(extracted_text)
                    score_list.append(val_acc)
                    test_acc_list.append(test_acc)
                    dt_list.append(dt)    
                    train_acc_list.append(train_acc)
                    all_status = 1
                    
                    for value in new_col:
                        if value == np.inf:
                            all_status = 0
                        elif value == -np.inf:
                            all_status = 0
                        elif np.isnan(value):
                            all_status = 0
                except Exception as e:
                    with open("err.log", "a") as f:
                        f.write(str(e))      
                        f.write("\n")
            else:
                pass
    

    np.save(f"../data/{name}/seed{seed}/score_list_step{args.step}.npy", score_list)
    np.save(f"../data/{name}/seed{seed}/test_acc_list_step{args.step}.npy", test_acc_list)
    np.save(f"../data/{name}/seed{seed}/train_acc_list_step{args.step}.npy", train_acc_list)
    with open (f"../data/{name}/seed{seed}/r_list_step{args.step}.json", "w") as json_file:
        json.dump(r_list, json_file)
    with open (f"../data/{name}/seed{seed}/dt_list_step{args.step}.json", "w") as json_file:
        json.dump(dt_list, json_file)   
        
    best_val = np.max(np.array(score_list))
    best_val_idx = np.where(np.array(score_list) == best_val)[0]
    if len(best_val_idx) == 1:
        best_test = np.max(np.array(test_acc_list)[best_val_idx])
    else:
        idx = np.where(np.array(train_acc_list)[best_val_idx] == np.min(np.array(train_acc_list)[best_val_idx]))[0]
        best_test = np.max(np.array(test_acc_list)[best_val_idx[idx]])
    
    print("Step {:.0f} | Val: {:.2f} | Test: {:.2f}".format(step+1, best_val*100, best_test*100))