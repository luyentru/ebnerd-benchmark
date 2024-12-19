import polars as pl
import datetime as dt

#============ preprocess data for time deltas============
def calculate_article_age(df, df_articles, max_value=1_000_000):
    """
    Add a new column to df_train containing normalized time deltas for all in-view articles.
    Negative deltas are set to 0, deltas are capped at max_value, and then normalized to [0, 1].
    
    Args:
        df_train (pl.DataFrame): DataFrame containing impression_time and inview_article_ids.
        df_articles (pl.DataFrame): DataFrame containing article_id and published_time.
        max_value (float): Maximum value for normalization (default is 1,000,000).

    Returns:
        pl.DataFrame: Updated df_train with a new column `article_age_normalized`.
    """
    # Create a mapping of article_id to published_time
    article_time_mapping = dict(
        zip(
            df_articles["article_id"].to_list(),
            df_articles["published_time"].to_list(),
        )
    )

    # Define a helper function to calculate and normalize deltas
    def compute_and_normalize_time_deltas(inview_article_ids, impression_time):
        deltas = []
        for article_id in inview_article_ids:
            published_time = article_time_mapping.get(article_id)
            if published_time:
                # Calculate delta in minutes
                delta = (impression_time - published_time).total_seconds() / 60
                # Set negative deltas to 0, cap at max_value, and normalize
                delta = min(max(delta, 0), max_value)
                normalized_delta = delta / max_value  # Normalize to [0, 1]
                deltas.append(normalized_delta)
            else:
                print("!!! Published time not found for articles:", article_id)
                deltas.append(None)  # Handle missing published_time
        return deltas

    assert all(
        isinstance(row, list) and len(row) > 0
        for row in df["article_ids_inview"].to_list()
    ), "Assertion Error: 'article_ids_inview' must be a non-empty list for every row."


    # Apply the function to compute and normalize time deltas
    df = df.with_columns(
        pl.struct(["article_ids_inview", "impression_time"]).apply(
            lambda row: compute_and_normalize_time_deltas(
                row["article_ids_inview"], row["impression_time"]
            )
        ).alias("article_age_normalized")
    )

    return df

#========================Mapping of published_time directly in DF Train==================================
def map_published_time(df, article_df):
    # Create a mapping of article_id to published_time
    article_time_mapping = dict(
        zip(
            article_df["article_id"].to_list(),
            article_df["published_time"].to_list()
        )
    )
    # Define a helper function to map published times while preserving order
    def map_article_times(article_ids):
        return [article_time_mapping.get(article_id, None) for article_id in article_ids]
    
    # Apply the mapping function to the `article_id_fixed` column using the helper
    df = df.with_columns(
        pl.col("article_id_fixed").apply(lambda article_ids: map_article_times(article_ids)).alias("published_time_list")
    )
    
    return df

#========================Calc average time difference for every user==================================
def calculate_average_time_difference(df):
    """
    Calculate the average time differences (impression time - published time) for each user, 
    cap at 500,000, and save them as a new column.

    Args:
        df (pl.DataFrame): The input DataFrame with impression times and published times.

    Returns:
        pl.DataFrame: Updated DataFrame with a new column containing the average time differences.
    """
    # Define a helper function to compute the average time difference for each user
    def compute_average_difference(impression_times, published_times):
        # Ensure the lists have the same length
        if len(impression_times) != len(published_times):
            raise ValueError("Impression times and published times must have the same length.")
        # Calculate the differences for each pair, skipping None values
        differences = [
            min((imp_time - pub_time).total_seconds() / 60, 500_000)
            if imp_time != dt.datetime(1970, 1, 1, 0, 0) and pub_time is not None
            else 0  # Assign 0 if invalid time
            for imp_time, pub_time in zip(impression_times, published_times)
        ]
        # Return the average of the differences
        return sum(differences) / len(differences) if differences else 0

    # Apply the helper function to compute the average difference for each row
    df = df.with_columns(
        pl.struct(["impression_time_fixed", "published_time_list"]).apply(
            lambda row: compute_average_difference(row["impression_time_fixed"], row["published_time_list"])
        ).alias("average_time_difference")
    )

    return df

#============== Calucate all time differences =======================
#========================Calc average recency score for every user==================================


def calculate_time_differences(df):
    """
    Calculate the time differences for each article, cap at 500,000, and save them in a list.

    Args:
        df (pl.DataFrame): The input DataFrame with impression times and published times.

    Returns:
        pl.DataFrame: Updated DataFrame with a new column containing the time differences as a list.
    """
    # Define a helper function to compute time differences for each article
    def compute_differences(impression_times, published_times):
        # Ensure the lists have the same length
        if len(impression_times) != len(published_times):
            raise ValueError("Impression times and published times must have the same length.")
        # Calculate the differences for each pair, skipping None values
        differences = [
            min((imp_time - pub_time).total_seconds() / 60, 500_000)
            if imp_time != dt.datetime(1970, 1, 1, 0, 0) and pub_time is not None
            else 0  # Assign 0 if invalid time
            for imp_time, pub_time in zip(impression_times, published_times)
        ]

        return differences

    # Apply the helper function to compute differences for each row
    df = df.with_columns(
        pl.struct(["impression_time_fixed", "published_time_list"]).apply(
            lambda row: compute_differences(row["impression_time_fixed"], row["published_time_list"])
        ).alias("time_differences")
    )

    return df


