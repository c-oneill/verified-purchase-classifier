import os
import random

'''
Data is organized into text files by product the reviews were pulled from.

This function takes the product files ( *_product_reviews.txt) and randomly
divides the reviews into three files:
    verifiedPurchase.devel
    verifiedPurchase.test
    verifiedPurchase.train

    split is 60/20/20, train/devel/test

each product file has the format (tab after label):
pictures (1 or 0), rating (out of 5), verified or unverified (u or v), then review text

URL
1 3 v	review text goes here
0 5 u	I really loved this product!!!
1 4 v	this is a terrible product

'''

productDirectory = "productFiles/"


def splitData():
    train = open("dataSplits/verifiedPurchase.train", 'w')
    devel = open("dataSplits/verifiedPurchase.devel", 'w')
    test = open("dataSplits/verifiedPurchase.test", 'w')

    for file in os.listdir(productDirectory):
        if file.endswith("_product_reviews.txt"):

            # iterate through lines
            with open(productDirectory + file) as f:
                next(f)
                for line in f:
                    assert "\t" in line

                    # randomly write to train, devel, or test
                    i = random.randint(0, 99)
                    if i < 60:
                        train.write(line)
                    elif i < 80:
                        devel.write(line)
                    elif i < 100:
                        test.write(line)

    train.close()
    devel.close()
    test.close()


def main():
    splitData()


if __name__ == '__main__':
    main()
