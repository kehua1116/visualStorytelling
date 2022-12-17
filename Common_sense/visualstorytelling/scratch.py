import dill as pickle


with open('../../data/vist/data/visualstorytelling/test.pkl','rb') as f:
    test_src = pickle.load(f)
    # print(list(test_src.keys())[:10])
    print(test_src["214418"]["target"])

with open('../../data/vist/data/visualstorytelling/new_test_story.pkl','rb') as f2:
    test_src = pickle.load(f2)
    print(test_src["214418"])