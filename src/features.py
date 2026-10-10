import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def ozellik_cikar(smiles):
    """Bir SMILES metnini sayısal özelliklere çevirir."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Geçersiz SMILES
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HDonors": Descriptors.NumHDonors(mol),
        "HAcceptors": Descriptors.NumHAcceptors(mol),
        "RingCount": Descriptors.RingCount(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        "TPSA": Descriptors.TPSA(mol),
    }


def main():
    df = pd.read_csv("data/delaney.csv")

    # Sadece SMILES ve hedef değişkeni alıyoruz
    df = df[["smiles", "measured log solubility in mols per litre"]]
    df.columns = ["smiles", "logS"]

    ozellikler = df["smiles"].apply(ozellik_cikar)
    gecerli = ozellikler.notna()
    df = df[gecerli].reset_index(drop=True)
    ozellik_df = pd.DataFrame(ozellikler[gecerli].tolist())

    sonuc = pd.concat([df, ozellik_df], axis=1)
    sonuc.to_csv("data/features.csv", index=False)

    print(sonuc.head())
    print(f"\nToplam molekül: {len(sonuc)}")


if __name__ == "__main__":
    main()