import os
import random

'''
Data is organized into text files by product the reviews were pulled from.

This function takes the product files and randomly
divides the reviews into three files:
    verifiedPurchase.devel
    verifiedPurchase.test
    verifiedPurchase.train

    split is 60/20/20, train/devel/test

data and format: https://s3.amazonaws.com/amazon-reviews-pds/readme.html

'''

productDirectory = "productFiles/"
maxDocs = 2000


def splitData():
    train = open("dataSplits/verifiedPurchase.train", 'w')
    devel = open("dataSplits/verifiedPurchase.devel", 'w')
    test = open("dataSplits/verifiedPurchase.test", 'w')

    numDocs = 0
    numUnverified = 0
    numVerified = 0

    for file in os.listdir(productDirectory):

        # iterate through lines
        with open(productDirectory + file) as f:
            next(f) # skip first line
            for line in f:

                label = line.split("\t")[11]

                # sometimes large number of verified purchases, need to filter some out
                if label == "N" and numUnverified <= numVerified:
                    numUnverified += 1
                elif label == "Y" and numVerified <= numUnverified:
                    numVerified += 1
                else:
                    continue

                # randomly write to train, devel, or test
                i = random.randint(0, 99)
                if i < 60:
                    train.write(line)
                elif i < 80:
                    devel.write(line)
                elif i < 100:
                    test.write(line)

                numDocs += 1
                if numDocs > maxDocs:
                    break

    train.close()
    devel.close()
    test.close()


def main():
    splitData()


if __name__ == '__main__':
    main()
