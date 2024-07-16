from tqdm import tqdm
import time
from colorama import Fore

# for i in tqdm(range(100), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.RED, Fore.RESET)):
#     time.sleep(0.1)

for i in tqdm(range(100), bar_format='{l_bar}%s{bar:30}%s{r_bar}{bar:-10b}'% (Fore.GREEN, Fore.RESET)):
    time.sleep(0.1)