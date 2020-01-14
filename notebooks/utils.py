from fastai import *
from fastai.text import *
from sklearn.metrics import roc_auc_score
from rdkit import Chem
import numpy as np
import threading

def randomize_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m,ans)
    return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True, kekuleSmiles=False)


def smiles_augmentation(df, N_rounds):
    dist_aug = {col_name: [] for col_name in df}

    for i in range(df.shape[0]):
        for j in range(N_rounds):
            dist_aug['SMILES'].append(randomize_smiles(df.iloc[i].SMILES))
            dist_aug['canonical'].append('no')

    df_aug = pd.DataFrame.from_dict(dist_aug)
    
    #merge with original df
    df = pd.concat([df, df_aug], sort=False).reset_index(drop=True)
    #shuffle the data
    df = df.reindex(np.random.permutation(df.index))
    return pd.DataFrame.from_dict(df).drop_duplicates('SMILES')

# Don't include the defalut specific token of fastai, only keep the padding token
BOS,EOS,FLD,UNK,PAD = 'xxbos','xxeos','xxfld','xxunk','xxpad'
TK_MAJ,TK_UP,TK_REP,TK_WREP = 'xxmaj','xxup','xxrep','xxwrep'
defaults.text_spec_tok = [PAD]

special_tokens = ['[BOS]', '[C@H]', '[C@@H]','[C@]', '[C@@]','[C-]','[C+]', '[c-]', '[c+]','[cH-]',
                   '[nH]', '[N+]', '[N-]', '[n+]', '[n-]' '[NH+]', '[NH2+]',
                   '[O-]', '[S+]', '[s+]', '[S-]', '[O+]', '[SH]', '[B-]','[BH2-]', '[BH3-]','[b-]',
                   '[PH]','[P+]', '[I+]', 
                  '[Si]','[SiH2]', '[Se]','[SeH]', '[se]', '[Se+]', '[se+]','[te]','[te+]', '[Te]']

class MolTokenizer(BaseTokenizer):
    def __init__(self, lang = 'en', special_tokens = special_tokens):
        self.lang = lang
        self.special_tokens = special_tokens
        
    def tokenizer(self, smiles):
        # add specific token '[BOS]' to represetences the start of SMILES
        smiles = '[BOS]' + smiles
        regex = '(\[[^\[\]]{1,10}\])'
        char_list = re.split(regex, smiles)
        tokens = []
        
        if self.special_tokens:
            for char in char_list:
                if char.startswith('['):
                    if char in special_tokens:
                        tokens.append(str(char))
                    else:
                        tokens.append('[UNK]')
                else:
                    chars = [unit for unit in char]
                    [tokens.append(i) for i in chars]                    
        
        if not self.special_tokens:
            for char in char_list:
                if char.startswith('['):
                    tokens.append(str(char))
                else:
                    chars = [unit for unit in char]
                    [tokens.append(i) for i in chars]
                
        #fix the 'Br' be splited into 'B' and 'r'
        if 'B' in tokens:
            for index, tok in enumerate(tokens):
                if tok == 'B':
                    if index < len(tokens)-1: # make sure 'B' is not the last character
                        if tokens[index+1] == 'r':
                            tokens[index: index+2] = [reduce(lambda i, j: i + j, tokens[index : index+2])]
        
        #fix the 'Cl' be splited into 'C' and 'l'
        if 'l' in tokens:
            for index, tok in enumerate(tokens):
                if tok == 'l':
                    if tokens[index-1] == 'C':
                            tokens[index-1: index+1] = [reduce(lambda i, j: i + j, tokens[index-1 : index+1])]
        return tokens    
    
    def add_special_cases(self, toks):
        pass


def auroc_score(input, target):
    input, target = input.cpu().numpy()[:,1], target.cpu().numpy()
    return roc_auc_score(target, input)

class AUROC(Callback):
    _order = -20 #Needs to run before the recorder

    def __init__(self, learn, **kwargs): self.learn = learn
    def on_train_begin(self, **kwargs): self.learn.recorder.add_metric_names(['AUROC'])
    def on_epoch_begin(self, **kwargs): self.output, self.target = [], []
    
    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            self.output.append(last_output)
            self.target.append(last_target)
                
    def on_epoch_end(self, last_metrics, **kwargs):
        if len(self.output) > 0:
            output = torch.cat(self.output)
            target = torch.cat(self.target)
            preds = F.softmax(output, dim=1)
            metric = auroc_score(preds, target)
            return add_metrics(last_metrics, [metric])



def test_get_scores(learn, ret=False):
    preds = learn.get_preds(ordered=True)
    print(f'Testing {len(preds[0])} molecues')
    p = torch.argmax(preds[0], dim=1)
    y = preds[1]
    tp = ((p + y) == 2).sum().item()
    tn = ((p + y) == 0).sum().item()
    fp = (p > y).sum().item()
    fn = (p < y).sum().item()
    cc = (float(tp)*tn - fp*fn) / np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    
    print(f'Accuracy: {(tp+tn)/len(y):.3f}')
    print(f'False Positives: {fp/len(y):.3f}')
    print(f'False Negatives: {fn/len(y):.3f}')
    print(f'Recall: {tp / (tp + fn):.3f}')
    print(f'Precision: {tp / (tp + fp):.3f}')
    print(f'Sensitivity: {tp / (tp + fn):.3f}')
    print(f'Specificity: {tn / (tn + fp):.3f}')
    print(f'MCC: {cc:.3f}')
    print(f'ROCAUC: {roc_auc_score(y,preds[0][:,1]):.3f}')

    if ret:
        return preds




