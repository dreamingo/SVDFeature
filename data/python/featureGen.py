import random

user_num = 943
feature_num = 10;
user_feature = open('../user_feature.dat', 'w')

for user in xrange(user_num):
    feature = ','.join([str(random.uniform(-1, 1))[0:4] for i in xrange(feature_num)])
    user_feature.write(str(user) + ',' + feature + '\n')


