import numpy as np
import pandas as pd
import copy
import logging
import math
import matplotlib.pyplot as plt

# Suppress matplotlib's debug output
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Configure the logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define log format with timestamp
    datefmt='%Y-%m-%d %H:%M:%S',  # Specify date and time format
    handlers=[
        logging.StreamHandler()  # Outputs to console
        # You can add more handlers, e.g., logging.FileHandler('logfile.log')
    ]
)

def ulcer_index(df: np.array) -> float:
    """
    Calculate the Ulcer Index of a time series.
    :param data: NumPy array of price data.
    :return: Ulcer Index.
    """
    max_price = np.maximum.accumulate(df)
    drawdown = (df - max_price) / max_price
    squared_drawdown = np.square(drawdown)
    mean_squared_drawdown = np.mean(squared_drawdown)
    ulcer_index = np.sqrt(mean_squared_drawdown)
    return ulcer_index
class GeneticAlgoMiner:
    def __init__(self,
        num_of_candles: int,
        pattern_size: int,
        population_size: int = 200,
        num_of_generation: int = 8,
        elitism_copys: int = 1,
        mutation_rate: float = 0.05,
        fresh_pattern_chance: float = 0.02,
        minimum_number_of_patterns: float = 0.025
    ):
        self._num_of_candles = num_of_candles
        self._pattern_size = pattern_size
        self._population_size = population_size
        self._num_of_generation = num_of_generation
        self._elitism_copys = elitism_copys
        self._mutation_rate = mutation_rate
        self._fresh_pattern_chance = fresh_pattern_chance
        self._minimum_number_of_patterns = minimum_number_of_patterns
        self._rules = []
        self._patterns = []
        self._reset_generation()
        self._selected_patterns = []
        self._selected_patterns_returns = []

    def train(self, df: pd.DataFrame):
        # Calculate Ulcer Index
        logging.info("Start Training Genetic Algorithm Miner")
        self._ulcer_index = ulcer_index(df['close'].to_numpy())

        # 1. Initialize patterns
        self._generate_random_pattern_rules()

        for i in range(self._num_of_generation):
            # Reset values for each generation
            self._reset_generation()

            # 2. Evaluate patterns with fitness functions
            self._evaluation(df)

            # 3. Elitism
            self._elite_selection()

            # 4. Parent Selection and Reproduction
            self._parent_selection_and_reproduction()
            
            # 5. Mutation
            # self._mutation()

            # 6. Repeat from step 2
            self._patterns = copy.deepcopy(self._next_generation_patterns)
            
            # 7. Generation result and reset
            logging.info(f"Generation {i + 1} Population Best Fitness {self._martin_ratios[0]} Population Average Fitness {np.mean(self._martin_ratios)}")
        # 8. Final results
        self._selected_patterns.append(self._patterns[0])
        self._selected_patterns_returns.append(self._returns[0])
        
        logging.info(f"Final Pattern: {self._patterns[0]}")
        logging.info(f"Total Returns: {self._total_returns[0]}, Profit Factor: {self._profit_factors[0]}")
        
        non_zero_elements = [x for x in self._returns[0] if x != 0]
        logging.info(f"Average Return: {np.mean(self._returns[0])}, Time In Market: {len(non_zero_elements) / len(df)}")
        logging.info(f"Martin Ratio: {self._martin_ratios[0]}, Ulcer Index: {self._ulcer_index}")
            
    def _reset_generation(self):
         # Reset values
        self._next_generation_patterns = []
        self._returns = [[] for _ in range(self._population_size)]

        # Initailize fitness values
        self._total_returns = [0] * self._population_size
        self._profit_factors = [0] * self._population_size
        self._martin_ratios = [0] * self._population_size

    def _generate_random_pattern_rules(self):
        self._rules = []
        self._patterns = []
        
        operators = ['<', '>']
        symbols = ['o', 'h', 'l', 'c']
        index = 0
        for i in range(self._num_of_candles):
            for j in range(self._num_of_candles):
                if i == j:
                    continue
            for symbol1 in symbols:
                for symbol2 in symbols:
                    for operator in operators:
                        index += 1
                        rule = f'{symbol1}[{i}] {operator} {symbol2}[{j}]'
                        self._rules.append(rule)
        # Generate random patterns
        if self._pattern_size > len(self._rules):
            raise ValueError("Pattern size cannot exceed the number of available rules.")
        for i in range(self._population_size):
            pattern = self._get_random_pattern(self._rules, self._pattern_size)
            self._patterns.append(pattern)

    def _calculate_fitness_functions(self):
        for i in range(self._population_size):
            returns_array = np.array(self._returns[i])
            self._total_returns[i] = max(sum(self._returns[i]), 0)

            positive_returns = abs(np.sum(returns_array > 0))
            negative_returns = abs(np.sum(returns_array < 0))
            self._profit_factors[i] = positive_returns / negative_returns if negative_returns != 0 else 0
            self._martin_ratios[i] = self._total_returns[i] / self._ulcer_index
        combined = list(zip(self._patterns, self._total_returns, self._profit_factors, self._martin_ratios))
        sorted_combined = sorted(combined, key=lambda x: x[3], reverse=True)
        self._patterns, self._total_returns, self._profit_factors, self._martin_ratios = zip(*sorted_combined)
        
    def _get_random_pattern(self, rules: list, pattern_size: int):
        rule_indices  = np.random.choice(len(rules), pattern_size, replace=False)
        return [rules[i] for i in rule_indices]

    def _evaluation(self, df: pd.DataFrame):

        """
        Evaluate each pattern on the given time series data.
        :param data: NumPy array of K bars (each entry contains 'open', 'high', 'low', 'close').
        """
        for index in range(len(self._patterns)):
            self._calculate_returns_with_single_pattern(index, df)
        self._calculate_fitness_functions()

    def _elite_selection(self):
        """
        Select the best patterns from the population.
        Add to next generation without any changes.
        """
        if self._elitism_copys == 0:
            return
        amount_of_elites = 0
        for i in range(self._population_size):
            if self._total_returns[i] > 0 and self._profit_factors[i] > 0 and self._martin_ratios[i] > 0:
                best_pattern = self._patterns[i]
                self._next_generation_patterns.append(best_pattern)
                amount_of_elites += 1
                
                if (amount_of_elites == self._elitism_copys):
                    break
            
    def _parent_selection_and_reproduction(self):
        while len(self._next_generation_patterns) < self._population_size:
            # Select parents
            parents = self._choose_parents()
            # logging.debug(f"Original Parents1: {parents[0]}")
            # logging.debug(f"Original Parents2: {parents[1]}")

            # Reproduce            
            split_point = np.random.choice(len(parents[0]) - 1) + 1
            
            pattern1_first = parents[0][:split_point]
            pattern1_second = parents[0][split_point:]
            pattern2_first = parents[1][:split_point]
            pattern2_second = parents[1][split_point:]

            parents[0] = pattern2_first + pattern1_second
            parents[1] = pattern1_first + pattern2_second
            
            # if (parents[0] in self._next_generation_patterns):
            #     print("Duplicate Parent: ", parents[0], len(self._next_generation_patterns))
            # if (parents[1] in self._next_generation_patterns):
            #     print("Duplicate Parent: ", parents[1], len(self._next_generation_patterns))
            
            if (len(self._next_generation_patterns) < self._population_size and parents[0] not in self._next_generation_patterns):
                self._next_generation_patterns.append(parents[0])
            if (len(self._next_generation_patterns) < self._population_size and parents[1] not in self._next_generation_patterns):
                self._next_generation_patterns.append(parents[1])
            
    def _choose_parents(self):
        parents = []
        num_of_parents = 2

        while len(parents) < num_of_parents:
            seed = np.random.uniform(0, sum(self._martin_ratios))
            parent_index = self._get_parent_index(seed)
            if (self._patterns[parent_index] in parents):
                # logging.debug("Duplicate Parent: ", "seed", seed, "parent_index", parent_index)
                continue
            
            parents.append(self._patterns[parent_index])
        return parents
            
    def _get_parent_index(self, seed: float):
        accumulated = 0
        for i in range(self._population_size):
            if (seed <= self._martin_ratios[i] + accumulated):
                return i
            accumulated += self._martin_ratios[i]

        return 0

    def _mutation(self):
        # Rule Adjustment
        num_of_rule_adjustments = int((self._mutation_rate - self._fresh_pattern_chance) * self._population_size)
        # logging.debug(f"Number of Rule Adjustments: {num_of_rule_adjustments}")
        adjustments_indices = np.random.choice(self._population_size, num_of_rule_adjustments, replace=False)
        for index in adjustments_indices:
            pattern = self._next_generation_patterns[index]
            # logging.debug("Adjusting pattern", pattern)
            # Choose symbol or index to adjust
            symbol_or_index = np.random.choice([0, 1])
            if symbol_or_index == 0:
                # Adjust symbol
                symbol_index = np.random.choice(len(pattern))
                symbol = pattern[symbol_index][0]
                if symbol == 'o':
                    new_symbol = np.random.choice(['h', 'l', 'c'])
                elif symbol == 'h':
                    new_symbol = np.random.choice(['o', 'l', 'c'])
                elif symbol == 'l':
                    new_symbol = np.random.choice(['o', 'h', 'c'])
                else:
                    new_symbol = np.random.choice(['o', 'h', 'l'])
                pattern[symbol_index] = new_symbol + pattern[symbol_index][1:]
            else:
                # Adjust index
                should_continue = True
                while should_continue:
                    candle_index = np.random.choice(self._num_of_candles)
                    rule_index = np.random.choice(self._pattern_size)
                    
                    # Split the string at the character you want to replace
                    current_pattern = pattern[rule_index]
                    
                    # Find the position of the bracketed index you want to change
                    split_position = current_pattern.find(']') - 1
                    if pattern[rule_index][2] == str(candle_index):
                        continue
                    else:
                        # Replace the part within the brackets with the new index
                        new_pattern = (
                            current_pattern[:split_position] +
                            str(candle_index) +
                            current_pattern[split_position + 1:]
                        )
                        pattern[rule_index] = new_pattern  # Update the pattern with the new string
                    should_continue = False

            # logging.debug("After Adjusted pattern", pattern)
        
        # Fresh Pattern Generation
        num_of_fresh_patterns = int(self._fresh_pattern_chance * self._population_size)
        # logging.debug(f"Number of Fresh Patterns: {num_of_fresh_patterns}")
        fresh_pattern_indices = np.random.choice(self._population_size, num_of_fresh_patterns, replace=False)
        for index in fresh_pattern_indices:
            self._next_generation_patterns[index] = self._get_random_pattern(self._rules, self._pattern_size)

    def _calculate_returns_with_single_pattern(self, pattern_index: int, df: pd.DataFrame):
        holding_period = 1
        pattern = self._patterns[pattern_index]
        upper_bound = len(df) - holding_period
        
        # Extract relevant columns only once
        close_prices = df['close'].values
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        for i in range(self._num_of_candles, upper_bound):
            if self.is_pattern_matching(pattern, i, open_prices, high_prices, low_prices, close_prices):
                next = close_prices[i + holding_period]
                current = close_prices[i]
                # score = (end_price - start_price) / start_price
                score = math.log(next / current)
                self._returns[pattern_index].append(score)
            
            else:
                self._returns[pattern_index].append(0)

    def is_pattern_matching(self, pattern: list, start_index, open_prices, high_prices, low_prices, close_prices):
        """
        Check if a pattern matches the data starting from a specific index.
        :param pattern: List of rules.
        :param start_index: Index to start checking from.
        :param open_prices: Open prices.
        :param high_prices: High prices.
        :param low_prices: Low prices.
        :param close_prices: Close prices.
        :return: True if the pattern matches the data, False otherwise.
        """
        for index, rule in enumerate(pattern):
            # Parse the rule
            symbol1, index1, operator, symbol2, index2 = self._parse_rule(rule)

            # Get values from data
            value1 = 0
            if (symbol1 == 'o'):
                value1 = open_prices[start_index - index1]
            elif (symbol1 == 'h'):
                value1 = high_prices[start_index - index1]
            elif (symbol1 == 'l'):
                value1 = low_prices[start_index - index1]
            else:
                value1 = close_prices[start_index - index1]
                
            value2 = 0
            if (symbol2 == 'o'):
                value2 = open_prices[start_index - index2]
            elif (symbol2 == 'h'):
                value2 = high_prices[start_index - index2]
            elif (symbol2 == 'l'):
                value2 = low_prices[start_index - index2]
            else:
                value2 = close_prices[start_index - index2]
            
            # Evaluate rule
            if not self._evaluate_rule(value1, operator, value2):
                return False
        return True

    def _parse_rule(self, rule: str):
        # Parses rule string into components
        elements = rule.split(' ')
        return elements[0][0], int(elements[0][2]), elements[1][0], elements[2][0], int(elements[2][2])

    def _symbol_to_symbol(self, symbol: str):
        # Maps symbols like 'o', 'h', 'l', 'c' to indices
        mapping = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'}
        return mapping[symbol]

    def _evaluate_rule(self, value1, operator, value2):
        # Evaluates a single rule
        if operator == '<':
            return value1 < value2
        elif operator == '>':
            return value1 > value2
        return False
    
    
if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = data[data.index < '01-01-2020']
    data = data[data.index > '12-31-2018']

    genetic_miner = GeneticAlgoMiner(3, 3)

    # Train the genetic miner 5 times to get 5 best patterns
    for i in range(5):
        genetic_miner.train(data)
        
    #plt.figure(figsize=(12, 6))
    # x = data['close'].to_numpy()
    # pd.Series(x).plot()
    for index, column in enumerate(genetic_miner._selected_patterns):
        label = ', '.join(column)
        x = [i for i in range(len(genetic_miner._selected_patterns_returns[index]))]
        cumulative_returns = np.cumsum(genetic_miner._selected_patterns_returns[index])
        plt.plot(x, cumulative_returns, label=label)
        
    
    plt.title("5 Patterns BTC-USDT 1H 2018 in Sample Cumulative Log Returns")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Log Returns")
    # plt.legend()
    plt.style.use('dark_background')  # Use a dark background style
    plt.grid()
    plt.show()