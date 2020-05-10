columns_to_change = set()
for col in telecom_data.select_dtypes(include=['float64']).columns:
    if((telecom_data[col].fillna(-9999) % 1  == 0).all()):
        columns_to_change.add(col)


print(len(columns_to_change))
columns_to_change.union(set(telecom_data.select_dtypes(include=['int64']).columns))
print(len(columns_to_change))

column_dict = dict()

for col in columns_to_change:
    max_value = telecom_data[col].max()
    min_value = telecom_data[col].min()
    load_type = "int64"
    if(min_value>= -128 and max_value<=127):
        load_type = 'int8'
    elif(min_value>= -32768 and max_value<=32767):
        load_type = 'int16'

    telecom_data[col] = telecom_data[col].fillna(-100)
    telecom_data[col] = telecom_data[col].astype(load_type)

print(column_dict)

telecom_data.info()