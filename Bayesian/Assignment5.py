import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as NormGauss
from data_utils import load_dataset

def sigmoid(z):
    return np.divide(1, (1+np.exp(-z)))

def LogLikelihood(h, y):
    return ( np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1 - h)))

def grad_LogLikelihood(h, y, x):
    return np.dot(x.T, (y - h))

def grad_grad_LogLikelihood(h, x):
    res = 0*np.eye(len(x[0]))
    for i in range(0, len(x)):
        loc_x = x[i,:]
        loc_x = np.reshape(loc_x,(len(loc_x),1))
        res += h[i]*(h[i]-1)*np.dot(loc_x, loc_x.T)
    return res

def logPrior(w, var):
    return -0.5 * ((len(w) * np.log(2*np.pi)) + (len(w) * np.log(var)) + (np.square(w).sum() / var))

def grad_logPrior(w, var):
    return -1 * w / var

def grad_grad_logPrior(var, M):
    return -1 / var * np.eye(M)

def accu(y_predicted, y_expected):
    correct = 0
    for x in range(len(y_predicted)):
        if (y_predicted[x] == y_expected[x]):
            correct += 1
    return (correct/len(y_predicted)) * 100

def predict(y, threshold = 0.5):
    return y >= threshold

def LaplaceApproximation(y, x, w, var):
    f_hat = sigmoid(np.dot(x,w))
    l_pyGwx = LogLikelihood(f_hat, y)# Probability of y given w and X
    l_pw = logPrior(w, var) # Probability of wx
    H = grad_grad_LogLikelihood(f_hat, x) + grad_grad_logPrior(var, len(w))
    l_gw = -0.5*(len(w)) * np.log(2*np.pi) + 0.5*(np.log(np.linalg.det(-1 * H)))
    return l_pyGwx + l_pw-l_gw

def BayesianGradientDecent(x_train, y_train, eta, var, w, iterations = 1000):
    k = 0
    f_hat = sigmoid(np.dot(x_train, w))
    GRAD = grad_logPrior(w, var) + grad_LogLikelihood(f_hat, y_train, x_train)
    w = w + eta * GRAD
    while GRAD.max() >  1e-10:
        f_hat = sigmoid(np.dot(x_train, w))
        GRAD = grad_logPrior(w, var) + grad_LogLikelihood(f_hat, y_train, x_train)
        w = w + eta * GRAD
        k += 1
    l_Pyx = LaplaceApproximation(y_train, x_train, w, var) # Return log marginal likelihood
    return l_Pyx, w, k

def Likelihood(h, y):
    return np.exp(LogLikelihood(h, y))

def PriorLogLikelihood(l_v, mu, sigma):
    l_ret = []
    for i in range(0, len(l_v)):
        v = l_v[i,:]
        v = np.reshape(v,(len(mu), 1))
        N = len(v)
        G = - np.log(2 * np.pi)*(N/2) -np.log(np.linalg.norm(sigma)**0.5 )
        G += ( - (0.5 * (v - mu).T.dot(np.linalg.inv(sigma)).dot( (v - mu) ) ) )[0,0]
        l_ret.append(G)
    l_ret = np.array(l_ret).reshape((len(l_v), 1))
    return l_ret

def Log_q_Probability(w, wMAP, H):
    return PriorLogLikelihood(w, wMAP, -np.linalg.inv(H))

def IS_Log_Likelihood(h, y):
    return np.dot(np.log(h), y) + np.dot(np.log(1-h), 1-y)

def ImportanceSamplingInference(wMAP, H, x_test, y_test, x_train, y_train, SampleSize = 3):
    # Sample SampleSize number of weights
    w_hat = np.random.multivariate_normal((wMAP.T)[0], -1 * np.linalg.inv(H), size = SampleSize)
    print(w_hat)
    mue = np.zeros((len(wMAP),1))
    log_prior_w_hat = PriorLogLikelihood(w_hat, mue, np.eye(len(wMAP)))
    log_q_distr = Log_q_Probability(w_hat, wMAP, H)
    log_likelihood_train = IS_Log_Likelihood(sigmoid(np.dot(w_hat, x_train.T)), y_train)
    r = log_likelihood_train + log_prior_w_hat - log_q_distr
    w_squiggle =  r - np.log(np.sum(np.exp(r)))
    y_likeli = np.zeros((len(y_test), 1))
    for i in range(0,len(w_hat)):
        f_hat = sigmoid(np.dot(x_test, (w_hat[i,:]).reshape(len(wMAP),1)))
        y_likeli += (f_hat * np.exp(w_squiggle[i, 0]))
        print(y_likeli)

    y_predict = predict(y_likeli)
    accuracy = accu(y_test, y_predict)
    print("Accuracy ", accuracy)
    print("Log Likelihood of", LogLikelihood(y_likeli, y_test)[0,0])

    for i in range(5):
        Plot_W = w_hat[:,i]

        plt.scatter(Plot_W, (np.exp(w_squiggle)), color='g', label = 'posterior')
        plt.scatter(Plot_W, (np.exp(log_q_distr)/np.sum(np.exp(log_q_distr))), color='m', label = 'proposal')
        plt.scatter(Plot_W, (np.exp(log_prior_w_hat)), color='r', label = 'prior')

        plt.scatter(Plot_W, (np.exp(r)), color='b', label = 'likelihood')
        plt.ylim([0, 0.015])
        plt.legend(loc='best')
        plt.xlabel("w[{}]".format(i))
        plt.ylabel('likelihood')
        plt.title('Posterior Likelihood  Visualization vs Weight {}'.format(i), fontsize=16, fontweight='bold')
        plt.grid(linestyle='-', linewidth=0.5)
        plt.savefig("PlotOfWeight{}.png".format(i))
        # plt.show()


def Metropolis_Hasting_MCMC(wOLD, x, y, Int_x, var):
    Burn = True
    l_predictionWeights = np.zeros((5,0))
    l_predictiveY = []

    l_epoch = []


    for i in range(0, 11000): # This Allows us to sample original 1,000 that will be burned and additional 10,000
        if i == 1000:
            print(wOLD)
            Burn = False
            #This Acts as a means of checking that we burned original 1,000
        if not Burn:
            u = np.log(np.random.rand())
            wNEW = proposal_sampler(wOLD, var)
            wNEW = np.reshape(wNEW, (5, 1))

            ua = np.minimum(1, (target_Distribution(wNEW, x, y) + proposal_pdf(wOLD, wNEW)) - target_Distribution(wOLD, x, y) - proposal_pdf(wNEW,wOLD))
            if u < ua:
                wOLD = wNEW # keep the proposal
            else:
                pass # reject the proposal
            if (i%100) == 0:
                l_predictionWeights = np.hstack([l_predictionWeights,wOLD])
                l_epoch.append(i - 1000)
                print(i, wNEW)


    mue = np.zeros((len(wOLD),1))
    l_predictionWeights = l_predictionWeights.T

    log_prior_w_hat = PriorLogLikelihood(l_predictionWeights, mue, np.eye(len(wOLD)))
    log_q_distr = Special_Log_q_Probability(l_predictionWeights, var)

    log_likelihood_train = IS_Log_Likelihood(sigmoid(np.dot(l_predictionWeights, x_train.T)), y_train)
    r = log_likelihood_train + log_prior_w_hat - log_q_distr

    w_squiggle =  r - np.log(np.sum(np.exp(r)))
    y_likeli = np.zeros((len(y_test), 1))
    l_predictiveY9 = []
    l_predictiveY10 = []
    for i in range(0,len(l_predictionWeights)):
        f_hat = sigmoid(np.dot(x_test, (l_predictionWeights[i,:]).reshape(len(wMAP),1)))
        y_likeli += (f_hat * np.exp(w_squiggle[i, 0]))
        l_predictiveY9.append(f_hat[8][0])
        l_predictiveY10.append(f_hat[9][0])

    y_predict = predict(y_likeli)
    accuracy = accu(y_test, y_predict)
    print("Accuracy ", accuracy)

    plt.figure(1)
    plt.title('Predictive Posterior for Flower 9', fontsize=16, fontweight='bold')
    plt.xlabel('Pr(y*|x*, w(i))')
    plt.xlim((0, 1))
    plt.ylabel('# Occurrences')
    plt.hist(l_predictiveY9, bins=5)
    plt.grid(linestyle='-', linewidth=0.5)
    plt.savefig('flower_9_bar.png')

    plt.figure(2)
    plt.title('Predictive Posterior for Flower 10', fontsize=16, fontweight='bold')
    plt.xlabel('Pr(y*|x*, w(i))')
    plt.xlim((0, 1))
    plt.ylabel('# Occurrences')
    plt.hist(l_predictiveY10, bins=5)
    plt.grid(linestyle='-', linewidth=0.5)
    plt.savefig('flower_10_bar.png')
    # plt.show()

def Special_Log_q_Probability(x, sigma):
    l_ret = []
    sigma = 1/sigma * np.eye(len(x[0]))
    for i in range(0, len(x)):
        if i == 0:
            mu = np.zeros((len(x[0]),1))
        else:
            mu = x[i-1,:]
            mu = np.reshape(mu,(len(x[0]), 1))
        v = x[i,:]
        v = np.reshape(v,(len(x[0]), 1))
        N = len(v)
        G = - np.log(2 * np.pi)*(N/2) -np.log(np.linalg.norm(sigma)**0.5 )
        G += ( - (0.5 * (v - mu).T.dot(np.linalg.inv(sigma)).dot( (v - mu) ) ) )[0,0]
        l_ret.append(G)
    l_ret = np.array(l_ret).reshape((len(x), 1))
    return l_ret

def proposal_pdf(a, b):
    a = a-b
    return (np.mean(-np.log(np.sqrt(2*np.pi)) - 0.5*np.square(a)))

def proposal_sampler(w0, var):
    M = len(w0)
    w0 = w0.T
    return np.random.multivariate_normal(w0[0], -np.eye(M)/var)

def target_Distribution(w, x, y):
    f = sigmoid(np.dot(x, w))
    posterior = y.T.dot(np.log(f)) + (1 - y).T.dot(np.log(1 - f))# - 0.5 * np.dot(w.T, w)
    return posterior


if __name__ == "__main__":
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    x_train, x_test, y_train, y_test = np.vstack((x_train, x_valid)), x_test, np.vstack((y_train[:,(1,)], y_valid[:,(1,)])), y_test[:,(1,)]
    y_train, y_test = 1 * y_train, 1 * y_test
    x_train, x_test = np.hstack([ np.ones((len(x_train), 1)), x_train]), np.hstack([ np.ones((len(x_test),  1)), x_test])


    Question1A = False
    Question1B = False
    Question1C = True

    if Question1A:
        print("--------------------------------------------------------------")
        print("Beginning Excution of Question 1 Part A")
        np.random.seed(43)
        l_var = [ 0.5, 1, 2]
        eta = 0.00005
        w = np.zeros((len(x_train[0]),1))
        for var in l_var:
            res, w, k = BayesianGradientDecent(x_train, y_train, eta, var, w)
            print("Log Marginal Likelihood is", res[0,0], "for var", var, 'that took ',k,'steps')
        print("Finished Excution of Question 1 Part A")
        print("--------------------------------------------------------------")

    if Question1B:
        print("--------------------------------------------------------------")
        print("Beginning Excution of Question 1 Part B")
        np.random.seed(6)
        sampleSize = 300
        eta = 0.00005
        w = np.zeros((len(x_train[0]),1))
        var = 10
        wMAP = np.array([[-0.878053],[0.29303],[-1.23477],[0.678156],[-0.894017]])
        f_hat = sigmoid(np.dot(x_train,wMAP))
        H = grad_grad_LogLikelihood(f_hat, x_train) + grad_grad_logPrior(var, len(wMAP))
        ImportanceSamplingInference(wMAP, H, x_test, y_test, x_train, y_train, sampleSize)

        print("Finished Excution of Question 1 Part B")
        print("--------------------------------------------------------------")

    if Question1C:
        print("--------------------------------------------------------------")
        print("Beginning Excution of Question 1 Part C")
        np.random.seed(6)
        perform_x, perform_y = x_test[[9,10]], y_test[[9,10]]

        wMAP = np.array([[-0.878053],[0.29303],[-1.23477],[0.678156],[-0.894017]])
        var = 2


        Metropolis_Hasting_MCMC(wMAP, x_test, y_test, perform_x, var)


        print("Finished Excution of Question 1 Part C")
        print("--------------------------------------------------------------")
