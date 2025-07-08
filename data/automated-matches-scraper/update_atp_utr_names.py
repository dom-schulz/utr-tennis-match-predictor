import pandas as pd

# Load data
matches_df = pd.read_csv("atp_utr_tennis_matches.csv")
profiles_df = pd.read_csv("profile_ids.csv")

# Create full name column in profiles
profiles_df["full_name"] = profiles_df["f_name"] + " " + profiles_df["l_name"]

# print(f"type of f_name: {type(profiles_df['f_name'])}")

# Create f_initial. last name column in profiles
profiles_df["f_initial_l_name"] = profiles_df["l_name"] + " " + profiles_df["f_name"].str[0] + "."

print(f'f initial l name: {profiles_df.head()}')


# Function to resolve a name using profile info
def replace_column(matches_df, profiles_df, columns_to_update):
    for col in columns_to_update:
        for index, row in matches_df.iterrows():
            if row[col] in profiles_df["f_initial_l_name"].values:
                matches_df.at[index, col] = profiles_df.loc[profiles_df["f_initial_l_name"] == row[col], "full_name"].values[0]
            else: 
                matches_df.at[index, col] = "NOT FOUND"
    print(f'matches_df: {matches_df.head()}')
    return matches_df

# Columns to update now (just winner)
columns_to_update = ["winner","p1","p2"]

# initialize return df with same columns as matches_df
return_df = matches_df.copy()

return_df = replace_column(matches_df, profiles_df, columns_to_update)

# Save the updated file
return_df.to_csv("atp_utr_matches_with_full_names.csv", index=False)

# Print num rows in return_df that have "NOT FOUND" regardless of column
num_not_found = 0
index_to_delete = []
for index, row in return_df.iterrows(): 
    
    delete = False
    for col in columns_to_update:
        
        if row[col] == "NOT FOUND" and delete == False:
            delete = True
            num_not_found += 1
            index_to_delete.append(index)
            continue

print(f'num rows in return_df that have "NOT FOUND": {num_not_found}')

# Drop rows that have "NOT FOUND"
return_df = return_df.drop(index_to_delete)

# Save the updated file
return_df.to_csv("atp_utr_matches_with_full_names.csv", index=False)