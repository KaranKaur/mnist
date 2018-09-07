from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# scale the images before applying PCA

def scale_transform(x_train, x_test):
    scaler = StandardScaler()

    # fit on training set
    scaler.fit(x_train)

    # apply transform to both train data and test data
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test


#def pca_method(val, x_train, x_test):
def pca_method(x_train, x_test):

    pca = PCA(0.55, svd_solver='full')
    #pca = PCA(val)
    pca.fit(x_train)

    #print('Number of PCA components are: %d for %f variance val' % (pca.n_components_, val))
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    return x_train, x_test
