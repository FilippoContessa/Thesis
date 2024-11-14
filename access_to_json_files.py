# dato un .json file, come faccio a ottenere un elemento specifico?
import pickle

with open("json_data.pkl", "rb") as f:
    json_files = pickle.load(f)


def get_vertebrae_coordinates(json_file_index, json_files, vertebrae_index):
    json_file = json_files[json_file_index]
    X , Y, Z  = json_file[vertebrae_index].get("X"), json_file[vertebrae_index].get("Y"), json_file[vertebrae_index].get("Z") 
    return X, Y, Z


def get_vertebrae_Y_distances(index, json_files):
    json_info = json_files[index]
    coordinate_Y_differences = []
    
    for i in range(1, len(json_info)-1):  # Ignora il primo e l'ultimo elemento per evitare errori
        y_current = json_info[i].get("Y")
        y_next = json_info[i + 1].get("Y")
    
        if y_current is not None and y_next is not None:
            # Calcola la differenza assoluta approssimando al valore intero più vicino per ciascuna dimensione
            diff_y = round(abs(y_next - y_current) / 2)
            
            label_current = json_info[i].get("label")
            label_next = json_info[i + 1].get("label")
            
            coordinate_Y_differences.append(diff_y)

    return coordinate_Y_differences
