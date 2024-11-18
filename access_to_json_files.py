# dato un .json file, come faccio a ottenere un elemento specifico?
import pickle

with open("training_json_data.pkl", "rb") as f:
    json_files = pickle.load(f)


def get_vertebrae_coordinates(json_file_index, json_files, vertebrae_index):
    json_file = json_files[json_file_index]
    X , Y, Z  = json_file[vertebrae_index].get("X"), json_file[vertebrae_index].get("Y"), json_file[vertebrae_index].get("Z") 
    return X, Y, Z


# TODO: implementalo con una singola funzione, prima verifica che sia effettivamente indispensabile.
def get_vertebrae_Y_prev_distances(index, json_files):
    json_info = json_files[index]
    coordinate_Y_prev_differences = []
    
    for i in range(1, len(json_info)):  # Inizia dal secondo elemento per evitare errori
        y_current = json_info[i].get("Y")
        y_prev = json_info[i - 1].get("Y")
        
        if y_current is not None and y_prev is not None:
            # Calcola la differenza assoluta approssimando al valore intero più vicino
            diff_y_prev = round(abs(y_current - y_prev) / 2)

            coordinate_Y_prev_differences.append(diff_y_prev)

    return coordinate_Y_prev_differences

def get_vertebrae_Y_next_distance(index, json_files):
    json_info = json_files[index]
    coordinate_Y_differences = []
    
    for i in range(1, len(json_info)-1):  # Ignora il primo e l'ultimo elemento per evitare errori
        y_current = json_info[i].get("Y")
        y_next = json_info[i + 1].get("Y")
        y_prev = json_info[i - 1].get("Y")
    
        if y_current is not None and y_next is not None:
            # Calcola la differenza assoluta approssimando al valore intero più vicino per ciascuna dimensione
            #diff_y_prev = round(abs(y_current - y_prev) / 2)
            diff_y_next = round(abs(y_next - y_current) / 2)
            
            label_current = json_info[i].get("label")
            label_next = json_info[i + 1].get("label")
            
            coordinate_Y_differences.append(diff_y_next)

    return coordinate_Y_differences
