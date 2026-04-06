import sys
import os
import psycopg2
import pandas as pd

def load_anomalies(data_root, db_name, db_user, socket_dir):
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        host=os.path.abspath(socket_dir)
    )
    cur = conn.cursor()

    # Expect structure: data_root/topic/final_output/country/anomalies.csv
    for topic in os.listdir(data_root):
        topic_path = os.path.join(data_root, topic)
        final_output_path = os.path.join(topic_path, "final_output")

        if not os.path.isdir(final_output_path):
            continue

        for country_code in os.listdir(final_output_path):
            csv_file = os.path.join(final_output_path, country_code, "anomalies.csv")
            if not os.path.isfile(csv_file):
                continue

            df = pd.read_csv(csv_file, parse_dates=["start", "end"])
            df = df.iloc[::-1]  # newest first

            for _, row in df.iterrows():
                start    = row["start"].date()
                end      = row["end"].date()
                score    = row["score"]
                residual = row["residual"]
                impact   = row["impact"]

                if not ((score == 0 and residual != 0) or (residual == 0 and score != 0)):
                    print(f"[SKIP] Invalid score/residual for {country_code} {start}")
                    continue

                cur.execute("""
                    SELECT end_date, terms
                      FROM anomalies
                     WHERE country_code = %s
                       AND start_date   = %s
                """, (country_code, start))
                row0 = cur.fetchone()

                if row0:
                    existing_end, existing_terms = row0
                    if existing_end == end and topic in existing_terms:
                        break

                    cur.execute("""
                        UPDATE anomalies
                           SET end_date = %s,
                               score = %s,
                               residual = %s,
                               impact = %s,
                               terms = CASE
                                   WHEN NOT (%s = ANY(terms))
                                   THEN array_append(terms, %s)
                                   ELSE terms
                               END
                         WHERE country_code = %s AND start_date = %s
                    """, (end, score, residual, impact, topic, topic, country_code, start))
                else:
                    cur.execute("""
                        INSERT INTO anomalies (
                            country_code, start_date, end_date,
                            score, residual, impact, terms, explanation_id
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s::text[], NULL)
                    """, (
                        country_code, start, end,
                        score, residual, impact, [topic]
                    ))

    conn.commit()
    cur.close()
    conn.close()
    print(f"[✓] All anomalies loaded from {data_root}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 load_anomalies.py <data_root> <db_name> <db_user> <socket_dir>")
        sys.exit(1)

    load_anomalies(*sys.argv[1:])
