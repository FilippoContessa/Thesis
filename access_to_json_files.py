# dato un .json file, come faccio a ottenere un elemento specifico?
import pickle

with open("json_data.pkl", "rb") as f:
    json_files = pickle.load(f)

# Carica il primo file JSON
json_info = json_files[0]

# Accede alla coordinata X,Y,Z della vertebra con label 17
#L'indice 1 identifica la vertebra che sto analizzando
vertebrae_index = 1
X , Y, Z  = json_info[vertebrae_index].get("X"), json_info[vertebrae_index].get("Y"), json_info[vertebrae_index].get("Z")  # Il secondo elemento ha la coordinata X pari a 94.8

print(f"Il valore della coordinata X è: {X}")
print(f"Il valore della coordinata Y è: {Y}")
print(f"Il valore della coordinata Z è: {Z}")