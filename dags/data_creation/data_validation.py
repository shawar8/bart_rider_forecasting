from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from data_creation.feature_store_variables import expectation_suite_name, features
from variables import target_column
import logging
logging.basicConfig(level=logging.INFO)


def build_expectations_suite():
    logging.info('Building Expectation Suite')
    expectation_suite = ExpectationSuite(
        expectation_suite_name= expectation_suite_name
    )
    expectation_suite.add_expectation(
        expectation_configuration= ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_ordered_list",
            kwargs= {'column_list': features}
        )
    )

    expectation_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={'column_list': ['date']}
        )
    )

    expectation_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={'column_list': ['data_input_time']}
        )
    )

    expectation_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={'column': 'month',
                    'value_set': [1,2,3,4,5,6,7,8,9,10,11,12]}
        )
    )

    expectation_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={'column': 'day',
                    'value_set': list(range(1, 32))}
        )
    )

    expectation_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={'column': 'day_of_week',
                    'value_set': list(range(7))}
        )
    )

    expectation_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={'column': 'is_weekend',
                    'value_set': [0,1]}
        )
    )

    expectation_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={'column': 'is_gd',
                    'value_set': [0, 1]}
        )
    )

    expectation_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={'column': 'is_hol',
                    'value_set': [0, 1]}
        )
    )

    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_min_to_be_between",
            kwargs={
                "column": "min_temp",
                "min_value": 0,
            },
        )
    )

    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_max_to_be_between",
            kwargs={
                "column": "max_temp",
                "max_value": 40,
            },
        )
    )

    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_min_to_be_between",
            kwargs={
                "column": target_column,
                "min_value": 0,
                "strict_min": True,
            },
        )
    )

    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": 'days_to_next_gd',
                "min_value": 0,
                "max_value": 270,
            },
        )
    )

    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": 'days_since_gd',
                "min_value": 0,
                "max_value": 270,
            },
        )
    )

    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": 'days_since_hol',
                "min_value": 0,
                "max_value": 110,
            },
        )
    )

    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": 'days_to_next_hol',
                "min_value": 0,
                "max_value": 110,
            },
        )
    )

    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": 'total_rain',
                "min_value": 0,
                "max_value": 110,
            },
        )
    )

    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": 'rain_duration',
                "min_value": 0,
                "max_value": 24,
            },
        )
    )

    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": 'num_passengers_two_week_avg',
                "min_value": 0,
                "strict_min": True,
            },
        )
    )
    logging.info('Completed Creating Expectations')
    return expectation_suite
