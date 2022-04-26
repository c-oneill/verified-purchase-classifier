import os
import random

'''
Data is organized into text files by product the reviews were pulled from.

This function takes the product files ( *_product_reviews.txt) and randomly
divides the reviews into three files:
    verifiedPurchase.devel
    verifiedPurchase.test
    verifiedPurchase.train

    split is 80/10/10, train/devel/test

each product file has the format (tab after label):
pictures (1 or 0), rating (out of 5), verified or unverified (u or v), TAB, review title, TAB, review text

URL
1 3 v	title   1 review text goes here
0 5 u	title2  I really loved this product!!!
1 4 v	EX  is a terrible product

'''

productDirectory = "productFiles/"


def splitData():
    train = open("dataSplits/verifiedPurchase.train", 'w')
    devel = open("dataSplits/verifiedPurchase.devel", 'w')
    test = open("dataSplits/verifiedPurchase.test", 'w')

    for file in os.listdir(productDirectory):

        # iterate through lines
        with open(productDirectory + file) as f:
            next(f) # skip first line
            for line in f:

                # randomly write to train, devel, or test
                i = random.randint(0, 99)
                if i < 80:
                    train.write(line)
                elif i < 90:
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
