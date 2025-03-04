import time
import random
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree
from sklearn import tree
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def evaluate_init(xtrain, ytrain, xval, yval, xtest, ytest, param_dict):
    clf = XGBClassifier(**param_dict, device="cuda", random_state=0, seed=0)
    clf.fit(xtrain, ytrain)
    xtrain_pred = clf.predict(xtrain)
    xval_pred = clf.predict(xval)
    xtest_pred = clf.predict(xtest)
    train_acc = accuracy_score(xtrain_pred, ytrain)
    val_acc = accuracy_score(xval_pred, yval)
    test_acc = accuracy_score(xtest_pred, ytest)
    return train_acc, val_acc, test_acc

def add_column(xtrain, xval, xtest, gen_c):
    gen_c = np.array(gen_c).reshape(-1,1)
    len_train, len_val = len(xtrain), len(xval)
    gen_c_train, gen_c_val, gen_c_test = gen_c[:len_train], gen_c[len_train:len_train + len_val], gen_c[len_train + len_val:]
    enc = MinMaxScaler()
    enc = enc.fit(gen_c_train)
    gen_c_train = enc.transform(gen_c_train)
    gen_c_val = enc.transform(gen_c_val)
    gen_c_test = enc.transform(gen_c_test)
    new_train = np.concatenate([xtrain, gen_c_train], axis = -1)
    new_val = np.concatenate([xval, gen_c_val], axis = -1)
    new_test = np.concatenate([xtest, gen_c_test], axis = -1) if xtest is not None else None
    return new_train, new_val, new_test

def evaluate(gen_c, xtrain, ytrain, xval, yval, xtest, ytest, param_dict):
    gen_c = np.array(gen_c).reshape(-1,1)
    len_train, len_val = len(xtrain), len(xval)
    gen_c_train, gen_c_val, gen_c_test = gen_c[:len_train], gen_c[len_train:len_train + len_val], gen_c[len_train + len_val:]
    enc = MinMaxScaler()
    enc = enc.fit(gen_c_train)
    gen_c_train = enc.transform(gen_c_train)
    gen_c_val = enc.transform(gen_c_val)
    gen_c_test = enc.transform(gen_c_test)
    new_train = np.concatenate([xtrain, gen_c_train], axis = -1)
    new_val = np.concatenate([xval, gen_c_val], axis = -1)
    new_test = np.concatenate([xtest, gen_c_test], axis = -1)
    clf = XGBClassifier(**param_dict, device="cuda", random_state=0, seed=0)
    clf.fit(new_train, ytrain)
    xtrain_pred = clf.predict(new_train)
    xval_pred = clf.predict(new_val)
    xtest_pred = clf.predict(new_test)
    train_acc = accuracy_score(xtrain_pred, ytrain)
    val_acc = accuracy_score(xval_pred, yval)
    test_acc = accuracy_score(xtest_pred, ytest)
    return train_acc, val_acc, test_acc

def get_cart(gen_c, xtrain, ytrain, xval, yval, seed):
    gen_c = np.array(gen_c).reshape(-1,1)
    len_train, len_val = len(xtrain), len(xval)
    gen_c_train, gen_c_val = gen_c[:len_train], gen_c[len_train:len_train + len_val]
    enc = MinMaxScaler()
    enc = enc.fit(gen_c_train)
    gen_c_train = enc.transform(gen_c_train)
    gen_c_val = enc.transform(gen_c_val)
    new_train = np.concatenate([xtrain, gen_c_train], axis = -1)
    new_val = np.concatenate([xval, gen_c_val], axis = -1)
    best_val = 0
    for j in range(1, 4):
        clf_CART = tree.DecisionTreeClassifier(max_depth = j, random_state = seed)
        clf_CART = clf_CART.fit(new_train, ytrain)
        xval_pred = clf_CART.predict(new_val)
        val_acc = accuracy_score(xval_pred, yval)
        if val_acc > best_val:
            best_val = val_acc
            best_CART = clf_CART
    return best_CART

def gen_prompt(r_list, dt_list, score_list, idx):

    s_l_np = np.array(score_list)
    sorted_idx = np.argsort(s_l_np)[-7:]
    new_r = []
    new_dt = []
    new_s = []
    for i in sorted_idx:
        new_r.append(r_list[i])
        new_dt.append(dt_list[i])
        new_s.append(score_list[i])
    
    text = f"I have some rules to generate x{idx} from x1"
    for i in range(1, idx-1):
        text += f", x{i+1}"
    text += ". We also have corresponding decision tree (CART) to predict 'y' from x1"
    for i in range(1, idx):
        text += f", x{i+1}"
    text += ". The rules are arragned in ascending order based on their scores evaluated with XGBoost classifier, where higher scores indicates better quality."
    text += "\n\n"
    for i in range(len(new_r)):
        text += f"Rule to generate x{idx}:\n{new_r[i]}\n"
        text += f"Decision tree (CART):\n{new_dt[i]}"
        text += "Score evaluated with XGBoost classifier:\n{:.0f}".format(new_s[i]*10000)
        text += "\n\n"
    text += f"Give me a new rule to generate x{idx} that is totally different from the old ones and has a score as high as possible. "
    text += f"Decision trees (including both CART and XGBoost classifier) trained with newly generated x{idx} should be better than the old ones. "
    text += f"Write the rule to generate x{idx} from x1"
    for i in range(1, idx-1):
        text += f", x{i+1}"
    text += f" in square brackets. Variables x1 ~ x{idx} are in [0, 1]. You can use various numpy function. Do not use np.log, np.sqrt, np.arcsin, np.arccos, np.arctan. Do not divide. When divide or using log, use (x+1) term. Think creatively. The new rule must be written with Python grammar."    
    text += f" Return the rule only with no explanation."
    return text

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth):
        indent = "  " * depth
        result = ""
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            result += f"{indent}if {name} > {threshold:.2f}:\n"
            result += recurse(tree_.children_right[node], depth + 1)
            result += f"{indent}else:\n"
            result += recurse(tree_.children_left[node], depth + 1)
        else:
            if tree_.value[node][0][0] > tree_.value[node][0][1]:
                result += f"{indent}y = 0.\n"
            else:
                result += f"{indent}y = 1.\n"
        
        return result

    return recurse(0, 0)

def load_model(model_path, peft_model_path=None):
    config = AutoConfig.from_pretrained(model_path)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        # attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if peft_model_path: model.load_adapter(peft_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def use_api(prompt, model, tokenizer, temp, iters=1):
    res = []

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            top_k=50,
            top_p=0.95,
            temperature=temp,
            num_return_sequences=iters,
        )
    prompt_length = inputs.input_ids.shape[1]
    for idx in range(output.shape[0]):
        res.append(tokenizer.decode(output[idx][prompt_length:], skip_special_tokens=True))
    return res