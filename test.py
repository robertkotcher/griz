import pandas as pd
import matplotlib.pyplot as plt

# Data 
data = {
    "difficultyTarget": {
        "0":402690497,"1":402690497,"2":402997206,"3":389508950,"4":392009692,
        "5":402904457,"6":389315112,"7":392962374,"8":391129783,"9":391203401
    },
    "timestamp":{
        "0":1515102747000,"1":1515688646000,"2":1467313104000,"3":1530414240000,
        "4":1520194941000,"5":1481051469000,"6":1530801978000,"7":1517100556000,
        "8":1522630178000,"9":1521516247000
    }
}

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Convert the timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit = 'ms')

# Plotting
df.plot(x ='timestamp', y='difficultyTarget', kind = 'line')
plt.savefig('/tmp/out.png')
