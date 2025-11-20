import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
import torch, re, os, pickle
from tqdm import tqdm

# === 数据清理 ===
def clean_sequence(sequence):
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', str(sequence).upper())

# === 基础特征 ===
def single_value_features(seq):
    analyzer = ProteinAnalysis(seq)
    return np.array([
        molecular_weight(seq, seq_type="protein"),
        IsoelectricPoint(seq).pi(),
        analyzer.aromaticity(),
        analyzer.instability_index(),
        analyzer.gravy()
    ])

def amino_acid_composition(seq):
    total=len(seq)
    return np.array([seq.count(aa)/total for aa in "ACDEFGHIKLMNPQRSTVWY"]) if total>0 else np.zeros(20)

def dipeptide_composition(seq):
    total=len(seq)-1
    dipeptides=[a+b for a in "ACDEFGHIKLMNPQRSTVWY" for b in "ACDEFGHIKLMNPQRSTVWY"]
    comp=Counter(seq[i:i+2] for i in range(total))
    return np.array([comp.get(dp,0)/total for dp in dipeptides]) if total>0 else np.zeros(400)

def physicochemical_properties(seq):
    total=len(seq)
    return np.array([
        sum(seq.count(a) for a in "KRH")/total,
        sum(seq.count(a) for a in "DE")/total,
        sum(seq.count(a) for a in "NQSTYC")/total,
        sum(seq.count(a) for a in "AVLIMFWPG")/total,
        total
    ]) if total>0 else np.zeros(5)

# === PseAAC 特征 ===
def pse_aac(seq, lambda_value=5, weight=0.05):
    seq = clean_sequence(seq)
    total = len(seq)
    if total == 0:
        return np.zeros(20 + lambda_value)

    hydrophobicity = {
        aa: val for aa, val in zip(
            "ACDEFGHIKLMNPQRSTVWY",
            [0.62,-0.90,0.29,-0.74,0.48,-0.40,-0.54,1.80,1.33,-1.20,-0.55,-0.70,
             -0.60,-0.76,0.12,0.07,-0.26,-0.57,-0.50,1.06]
        )
    }

    aac = amino_acid_composition(seq)
    correlation = []

    for lag in range(1, lambda_value + 1):
        if total - lag <= 0:  # 防止除零
            correlation.append(0.0)
        else:
            autocorr = 0.0
            for i in range(total - lag):
                autocorr += hydrophobicity[seq[i]] * hydrophobicity[seq[i + lag]]
            correlation.append(autocorr / (total - lag))

    # 归一化调整
    denominator = (1 - weight) + weight * np.sum(correlation)
    if denominator == 0:
        aac_adjusted = aac
    else:
        aac_adjusted = (aac * (1 - weight) / denominator)

    return np.concatenate([aac_adjusted, correlation])


# === CTD 特征 ===
def ctd_features(seq):
    seq=clean_sequence(seq)
    if len(seq)==0:
        return np.zeros(21)

    groups = {
        'polar': set("NQSTYC"),
        'neutral': set("DG"),
        'nonpolar': set("AVLIMFWPHKRE")
    }
    comp = [sum(aa in g for aa in seq) / len(seq) for g in groups.values()]

    transitions=[]; keys=list(groups.keys())
    for i in range(len(seq)-1):
        for k1 in keys:
            for k2 in keys:
                if k1!=k2 and seq[i] in groups[k1] and seq[i+1] in groups[k2]:
                    transitions.append(f"{k1}-{k2}")
    transition_counts = Counter(transitions)
    trans_feat = [
        transition_counts.get(f"{keys[i]}-{keys[j]}",0)/(len(seq)-1)
        for i in range(len(keys)) for j in range(len(keys)) if i!=j
    ]

    dist_feat=[]
    for g in groups.values():
        pos=[i for i,a in enumerate(seq) if a in g]
        if pos:
            for p in [0.25,0.50,0.75,1.0]:
                idx = int(len(pos)*p) - 1 if p == 1.0 else int(len(pos)*p)
                idx = max(0, min(idx, len(pos)-1))  # 保证不越界
                dist_feat.append(pos[idx]/len(seq))
        else:
            dist_feat.extend([0,0,0,0])

    return np.concatenate([comp, trans_feat, dist_feat])

# === ProtBert 批量嵌入 ===
def protbert_embedding_batch(sequences, model_path, batch_size=32, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path,"rb") as f: return pickle.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings=[]
    for i in tqdm(range(0,len(sequences),batch_size), desc="ProtBert Embedding"):
        batch_seq = [" ".join(list(seq)) for seq in sequences[i:i+batch_size]]
        inputs=tokenizer(batch_seq, return_tensors="pt", padding=True, truncation=True, max_length=50)
        inputs={k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs=model(**inputs)
        batch_emb=outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_emb)
    embeddings=np.array(embeddings)
    if cache_path:
        with open(cache_path,"wb") as f: pickle.dump(embeddings,f)
    return embeddings

# === 特征融合 ===
def extract_features_df(df, model_path, sequence_col="sequence", label_col="label", cache_id=None):
    sequences = [clean_sequence(seq) for seq in df[sequence_col]]
    cache_path = f"protbert_cache_{cache_id}.pkl" if cache_id else None
    protbert_features = protbert_embedding_batch(sequences, model_path, batch_size=16, cache_path=cache_path)
    all_features = []
    for seq, protbert_feat in zip(sequences, protbert_features):
        basic_feat = np.concatenate([
            single_value_features(seq),
            amino_acid_composition(seq),
            dipeptide_composition(seq),
            physicochemical_properties(seq),
            pse_aac(seq),
            ctd_features(seq)
        ])
        all_features.append(np.concatenate([basic_feat, protbert_feat]))
    feature_df = pd.DataFrame(all_features)
    if label_col in df.columns:
        feature_df[label_col] = df[label_col].values
    return feature_df

# === 主入口 ===
def process_dataset(input_path, model_path, output_path, sheet_name=None):
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
        cache_id = os.path.basename(input_path).split(".")[0]
    elif input_path.endswith(".xlsx"):
        df = pd.read_excel(input_path, sheet_name=sheet_name)
        cache_id = f"{os.path.basename(input_path)}_{sheet_name}"
    else:
        raise ValueError("Unsupported file type")
    # 自动匹配列名
    features_df = extract_features_df(df, model_path, sequence_col="sequence", label_col="label", cache_id=cache_id)
    features_df.to_csv(output_path, index=False)
    print(f"Feature extraction complete. Saved to {output_path}")

# === 运行示例 ===
if __name__ == "__main__":
    process_dataset(
        input_path="/home/zqlibinyu/prediction/data/Alternate_DatasetB.xlsx",
        model_path="/home/zqlibinyu/prediction/prot_bert/",
        output_path="/home/zqlibinyu/prediction/data/output/Alternate_DatasetB_sheet1.csv",
        sheet_name="Sheet1"  # Excel里指定sheet
    )
