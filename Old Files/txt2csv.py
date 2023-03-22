import re
import pandas as pd


def main():
    # Defining the output dataframe
    column_names = ['id', 'genre', 'subgenre', 'artist', 'title']
    out_df = pd.DataFrame(columns=column_names)
    temp_list = []

    with open("./Contents of SLAC/SLAC_text_inventory.txt") as f:
        lines = f.readlines()

    for x in lines:
        x = x.replace("\n", "")
        song = [x] + re.split(r"\s-\s|/", x)
        temp_list.append(dict(zip(column_names, song)))

    out_df = pd.concat([out_df, pd.DataFrame(temp_list)], axis=0, ignore_index=True)
    out_df.to_csv('./Contents of SLAC/SLAC_csv_inventory.csv')


if __name__ == "__main__":
    main()
