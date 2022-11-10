import pandas as pd

def normalize_geometry(geometry):
    geometry = ', ' +geometry[1:-1]
    tokens = geometry[1:-1].split(',')
    # LINESTRING (30 10, 10 30, 40 40)
    wkt = 'LINESTRING('
    for index, token in enumerate(tokens):
        pos = token.split(' ')
        if index< len(tokens) - 1:
            lon = float(pos[1]) / 10000000.0
            lat = float(pos[2]) / 10000000.0
        else:
            lon = float(pos[1]) / 10000000.0
            lat = float(pos[2]) / 1000000.0
        wkt += f'{lon} {lat},'
    wkt = wkt[0:-1] + ')'
    return wkt


def get_value(attr, value, tab, index):
    new_value =''.join(i for i in str(value) if i.isdigit())
    if index < 4:
        tab[index] = new_value
    return tab


if __name__ == '__main__':
    files = ['./data/resultNoViol.csv','./data/result.csv']
    df3 = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, sep=';')
        # RequestId;FeatureID;attribute name;attribute value;feature geometry;violations state
        df = df.reset_index()  # make sure indexes pair with number of rows

        df2 =  pd.DataFrame(columns=['feature_id', 'left_min', 'left_max', 'right_min', 'right_max', 'geometry', 'is_violation'])
        init = df.loc[0]['RequestId']
        val = [0,0,0,0]
        feature_id = ''
        geometry = ''
        state = ''
        id = 0

        for index, row in df.iterrows():
            if init == row['RequestId']:
                val = get_value(row['attribute name'], row['attribute value'], val, id)
                feature_id = row["FeatureID"]
                geometry = row["feature geometry"]
                state = row["violations state"]
                id += 1
            else:
                df2.loc[len(df2)] = [feature_id, val[0], val[1], val[2], val[3], normalize_geometry(geometry), state]
                init = row['RequestId']
                val = [0,0,0,0]
                feature_id = ''
                geometry = ''
                state = ''
                val = get_value(row['attribute name'], row['attribute value'], val, 0)
                id = 1

        df2 = df2.drop_duplicates()
        df2.to_csv(f'{file}_', index=False)

        df3 = pd.concat([df2, df3])
    df3.to_csv('./data/all.csv', index=False)
