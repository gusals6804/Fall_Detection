from pymongo import MongoClient
import datetime

host = "223.194.70.104"
port = "33017"
my_client = MongoClient(host, int(port))
print(my_client)

mydb = my_client['video']
mycol = mydb['fall']

list = mycol.find()

for x in list:
    print(x)

print(my_client.list_database_names())

