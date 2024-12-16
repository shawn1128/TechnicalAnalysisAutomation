import numpy as np
import pandas as pd
import copy

def ulcer_index(data: np.array) -> float:
    """
    Calculate the Ulcer Index of a time series.
    :param data: NumPy array of price data.
    :return: Ulcer Index.
    """
    max_price = np.maximum.accumulate(data)
    drawdown = (data - max_price) / max_price
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
        self._mutation_rate = mutation_rate
        self._fresh_pattern_chance = fresh_pattern_chance
        self._minimum_number_of_patterns = minimum_number_of_patterns
        self._rules = []
        self._patterns = []
        self._next_generation_patterns = []
        self._returns = [[] for _ in range(self._population_size)]

        # Initailize fitness values
        self._total_returns = [0] * self._population_size
        self._profit_factors = [0] * self._population_size
        self._martin_ratios = [0] * self._population_size
        return

    def _generate_random_pattern_rules(self):
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
        
    def train(self, data: pd.DataFrame):
        # Calculate Ulcer Index
        self._ulcer_index = ulcer_index(data['close'].to_numpy())

        # 1. Initialize patterns
        self._generate_random_pattern_rules()

        for i in range(self._num_of_generation):
            # Reset values
            self._next_generation_patterns = []
            self._returns = [[] for _ in range(self._population_size)]

            # Initailize fitness values
            self._total_returns = [0] * self._population_size
            self._profit_factors = [0] * self._population_size
            self._martin_ratios = [0] * self._population_size

            # 2. Evaluate patterns with fitness functions
            self._evaluation(data)

            # 3. Elitism
            best_pattern = self._elite_selection()
            self._next_generation_patterns.append(best_pattern)

            # 4. Parent Selection and Reproduction
            self._parent_selection_and_reproduction()
            
            # 5. Mutation
            self._mutation()

            # 6. Repeat from step 2
            self._patterns = copy.deepcopy(self._next_generation_patterns)
            
            # 7. Generation result and reset
            print("Generation", i, "Population Best Fitness", self._martin_ratios[0], " Population Average Fitness", np.mean(self._martin_ratios))
        print ("Final Pattern", self._patterns[0])
        for i in range(5):
            print("Final Pattern ", i)
            print(self._patterns[i])
            print("Total Returns ", self._total_returns[i], "Profit Factor ", self._profit_factors[i])
            print("Average Return ", np.mean(self._returns[i]), "Time In Market ", len(self._returns[i]) / len(data))
            print("Martin Ratio ", self._martin_ratios[i], "Ulcer Index ", self._ulcer_index)
            

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

    def _evaluation(self, data: pd.DataFrame):

        """
        Evaluate each pattern on the given time series data.
        :param data: NumPy array of K bars (each entry contains 'open', 'high', 'low', 'close').
        """
        for index, pattern in enumerate(self._patterns):
            # print("calculating pattern", index)
            self._calculate_returns_with_single_pattern(pattern, index, data)
        self._calculate_fitness_functions()

    def _elite_selection(self):
        """
        Select the best patterns from the population.
        :return: List of best patterns.
        """
        max_index = 0
        for i in range(self._population_size):
            if self._total_returns[i] <= 0 or self._profit_factors[i] <= 0 or self._martin_ratios[i] <= 0:
                continue
            else:
                max_index = i
                break

        return self._patterns[max_index]
            
    def _parent_selection_and_reproduction(self):
        while len(self._next_generation_patterns) < self._population_size:
            # Select parents
            parents = self._choose_parents()
            # print("Original Parents", parents)

            # Reproduce
            rule1 = str(parents[0][0])
            rule2 = str(parents[1][0])
            
            parents[0][0] = rule2
            parents[1][0] = rule1
            # print("New Parents", parents)
            
            if (len(self._next_generation_patterns) < self._population_size):
                self._next_generation_patterns.append(parents[0])
            if (len(self._next_generation_patterns) < self._population_size):
                self._next_generation_patterns.append(parents[1])
            # print("End of Reproduction")
            
    def _choose_parents(self):
        parents = []
        num_of_parents = 2

        while len(parents) < num_of_parents:
            seed = np.random.uniform(0, sum(self._martin_ratios))
            parent_index = self._get_parent_index(seed)
            if (self._patterns[parent_index] in parents):
                # print("Duplicate Parent: ", "seed", seed, "parent_index", parent_index)
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
        adjustments_indices = np.random.choice(self._population_size, num_of_rule_adjustments, replace=False)
        for index in adjustments_indices:
            pattern = self._next_generation_patterns[index]
            # print("Adjusting pattern", pattern)
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
                        # print('Same index')
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

            # print("After Adjusted pattern", pattern)
        
        # Fresh Pattern Generation
        num_of_fresh_patterns = int(self._fresh_pattern_chance * self._population_size)
        fresh_pattern_indices = np.random.choice(self._population_size, num_of_fresh_patterns, replace=False)
        for index in fresh_pattern_indices:
            self._next_generation_patterns[index] = self._get_random_pattern(self._rules, self._pattern_size)

    def _calculate_returns_with_single_pattern(self, pattern: list, patternIndex: int, data: pd.DataFrame):
        holding_period = 1
        for i in range(self._num_of_candles, len(data) - holding_period):
            if self.is_pattern_matching(data, pattern, i):
                end_price = data.iloc[i + holding_period]['close']
                start_price = data.iloc[i]['close']
                score = (end_price - start_price) / start_price
                self._returns[patternIndex].append(score)
            
            else:
                self._returns[patternIndex].append(0)

    def is_pattern_matching(self, data: pd.DataFrame, pattern: list, start_index):
        """
        Check if a pattern matches the data starting from a specific index.
        :param data: NumPy array of K bars.
        :param pattern: List of rules for a pattern.
        :param start_index: Starting index for matching.
        :return: Boolean indicating if pattern matches.
        """
        for index, rule in enumerate(pattern):
            # Parse the rule
            symbol1, index1, operator, symbol2, index2 = self._parse_rule(rule)

            # Get values from data
            value1 = data.iloc[start_index - index1][self._symbol_to_symbol(symbol1)]
            value2 = data.iloc[start_index - index2][self._symbol_to_symbol(symbol2)]
            
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
    data = data[data.index < '01-01-2019']
    data = data[data.index > '12-31-2017']

    genetic_miner = GeneticAlgoMiner(3, 2)
    genetic_miner.train(data)