import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from apyori import apriori


class DataProcessor:

    def load_dataframe(self, path_to_csv:str) -> pd.DataFrame:
        """
        loads the dataframe from a csv file
        :param path_to_csv: string describing the path to the csv file from the current working directory
        :return: pandas dataframe of the dataset
        """
        assert os.path.exists(path_to_csv)
        return pd.read_csv(path_to_csv)

    def sample_dataset(self, df: pd.DataFrame, labels: list, sample_frac, seed=42) -> pd.DataFrame:
        """
        Groups data in a dataframe by the given labels and then samples accordingly to the desired fraction
        :param df: pandas dataframe
        :param labels: list of labels to group
        :param sample_frac: fraction of data to sample
        :param seed: random seed for the sampling
        :return: pandas dataframe of the sampled data
        """
        return df.groupby(labels).sample(frac=sample_frac, random_state=seed)

    def remove_infrequent_items(self, df: pd.DataFrame, threshold=70) -> pd.DataFrame:
        """
        removes items that are not present above the given threshold from the given dataframe
        :param df: pandas dataframe
        :param threshold: threshold to remove items
        :return: pruned dataframe
        """
        counts = df["product_id"].value_counts()
        to_remove = counts[counts <= threshold].index
        return df[~df["product_id"].isin(to_remove)]
    
    def append_product_list_to_orders(self, orders_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
        """
        groups products that belong to the same order by id in a list, then adds that list to the corresponding order entry
        in the dataframe containing the orders
        :param orders_df: dataframe containing the orders
        :param products_df: dataframe containing the products with their corresponding order id
        :return: dataframe containing the orders with an extra "products" column containing the orders products
        """
        # sort the items in the order they were added to the cart, then covert to lists
        order_items = (
            products_df.sort_values(["order_id", "add_to_cart_order"]).groupby("order_id")["product_id"].apply(
                list).reset_index(name="products"))

        # add each product list to their respective order in a new "products" column
        return orders_df.merge(order_items, on="order_id", how="inner")

    def get_association_rules(self, transactions: list, min_support, min_confidence):
        """
        runs the a priori algorithm on a list of transactions to find the association rules
        :param transactions: list of transactions, every transaction is a list of product ids
        :param min_support: minimum support for an itemset to be considered frequent
        :param min_confidence: minimum confidence for the apriori algorithm to accept the rules
        :return: a list of association rules
        """

        rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)
        return list(rules)

    def print_association_rules(self, rules, product_map: dict):
        """
        prints the association rules in a readable format using the product name instead of identifier, also including
        the support and confidence of the rules
        :param rules: list of association rules
        :param product_map: dictionary mapping product ids to product names
        """
        i = 0
        for rule in rules:
            for stat in rule.ordered_statistics:
                if len(stat.items_base) > 0:
                    base_names = [product_map.get(pid, str(pid)) for pid in stat.items_base]
                    add_names = [product_map.get(pid, str(pid)) for pid in stat.items_add]
                    print(f"{i}: {base_names} -> {add_names}, support: {rule.support :0.3}, confidence: {stat.confidence :0.3}")
                    i += 1
        print(f"rule count: {i}")

    def hour_token(self, hour):
        """
        turns an hour of the day to an integer representation of the parts of the day as follows:
        5-12: Morning = 0
        12-17: Afternoon = -1
        17-21: Evening = -2
        21-5: Night = -3
        :param hour: hour of the day to convert
        :return: integer representation of the parts of the day
        """
        if 5 <= hour < 12:
            return 0
        elif 12 <= hour < 17:
            return -1
        elif 17 <= hour < 21:
            return -2
        else:
            return -3

    def add_time_tokens(self, orders_df: pd.DataFrame, product_map: dict):
        """
        adds time tokens to the orders transaction list based on when the order was placed
        also adds the token mapping from integer to part of the day in the product map 
        :param orders_df: dataframe containing the orders
        :param product_map: dictionary mapping product ids to product names
        :return:
        """
        orders_df["products"] = orders_df.apply(lambda row: row["products"] + [self.hour_token(row["order_hour_of_day"])], axis=1)
        for i, token in enumerate(["MORNING", "AFTERNOON", "EVENING", "NIGHT"]):
            product_map[-i] = token

    def plot_min_support_rule_count(self, transactions:list, start, end, step_size):
        """
        plots the association rule count when running the a priori algorithm for the given transactions at various
        different minimum support thresholds in the specified interval with minimum confidence set to 0
        :param transactions: list of transactions, every transaction is a list of products ids
        :param start: start of the interval
        :param end: end of the interval
        :param step_size: step size of the interval
        """
        min_supports = []
        rule_counts = []

        min_support = start
        while min_support <= end:
            results = self.get_association_rules(transactions, min_support=min_support, min_confidence=0)
            count = 0
            for rule in results:
                for stat in rule.ordered_statistics:
                    if len(stat.items_base) > 0:
                        count += 1

            min_supports.append(min_support)
            rule_counts.append(count)
            min_support += step_size

        plt.figure()
        plt.plot(min_supports, rule_counts, marker="o")
        plt.xlabel("Minimum Support")
        plt.ylabel("Number of Association Rules")
        plt.title("Effect of Minimum Support on Rule Count")
        plt.grid(True)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(step_size))
        plt.xticks(rotation=45)
        plt.show()

    def plot_min_confidence_rule_count(self, transactions:list, min_support, start, end, step_size):
        """
        plots the association rule count when running the a priori algorithm for the given transactions at various
        different minimum confidence thresholds in the specified interval with the specified minimum support
        :param transactions: list of transactions, every transaction is a list of products ids
        :param start: start of the interval
        :param end: end of the interval
        :param step_size: step size of the interval
        """
        min_confidences = []
        rule_counts = []

        min_confidence = start
        while min_confidence <= end:
            results = self.get_association_rules(transactions, min_support=min_support, min_confidence=min_confidence)
            count = 0
            for rule in results:
                for stat in rule.ordered_statistics:
                    if len(stat.items_base) > 0:
                        count += 1

            min_confidences.append(min_confidence)
            rule_counts.append(count)
            min_confidence += step_size

        plt.figure()
        plt.plot(min_confidences, rule_counts, marker="o")
        plt.xlabel("Minimum Confidence")
        plt.ylabel("Number of Association Rules")
        plt.title(f"Effect of Minimum Confidence on Rule Count (minimum support={min_support})")
        plt.grid(True)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(step_size))
        plt.xticks(rotation=45)
        plt.show()


def main():
    # load the dataframes for the orders, order products and product information
    dataprocessor = DataProcessor()
    orders_df = dataprocessor.load_dataframe("data\\orders.csv")
    products_df = dataprocessor.load_dataframe("data\\order_products.csv")
    products_info_df = dataprocessor.load_dataframe("data\\products.csv")


    # sample 10% of the orders dataset
    orders_df = dataprocessor.sample_dataset(orders_df, ["order_dow", "order_hour_of_day"], sample_frac=0.1)

    # remove product order entries if the order is no longer present in the sampled orders dataset
    products_df = products_df[products_df["order_id"].isin(orders_df["order_id"])]

    # remove the products that appear infrequently in the dataset
    products_df = dataprocessor.remove_infrequent_items(products_df)

    # add a list containing the product ids of an order to its entry in a new column in the orders dataframe
    orders_df = dataprocessor.append_product_list_to_orders(orders_df, products_df)


    # create a dictionary mapping the product ids to their names
    product_map = dict(zip(products_info_df["product_id"], products_info_df["product_name"]))

    # UNCOMMENT THE FOLLOWING LINE OF CODE TO OBTAIN THE ASSOCIATION RULES USED IN TASK 2B: PURCHASE TIMING
    # dataprocessor.add_time_tokens(orders_df, product_map)

    # create a list of the products list of every order
    transactions = orders_df["products"].tolist()

    # compute the association rules using the a priori algorithm
    rules = dataprocessor.get_association_rules(transactions, min_support=0.001, min_confidence=0.3)

    # print the found association rules
    dataprocessor.print_association_rules(rules, product_map)


    # plots a graph showing the association rule counts at different minimum support thresholds
    # dataprocessor.plot_min_support_rule_count(transactions, 0.001, 0.020, 0.001)

    # plots a graph showing the association rule counts at different minimum confidence thresholds
    # dataprocessor.plot_min_confidence_rule_count(transactions, 0.001, 0.1, 0.5, 0.1)

if __name__ == "__main__":
    main()


