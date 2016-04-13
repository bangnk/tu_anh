cd ~/git
git init --bare tu_anh.git
cd ~/envs
git clone bang@mac.local:/Users/bang/git/tu_anh.git
mkvirtualenv tu_anh
cdvirtualenv
pip install -r ./requirements.txt