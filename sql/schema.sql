
CREATE TABLE explanations (
    id               SERIAL PRIMARY KEY,

    summary          TEXT    NOT NULL,

    links            TEXT[],                       -- array of URLs
    tags             TEXT[],                       -- array of event tags

    cause            TEXT,                         -- only when tags contains 'censorship'
    affected_regions TEXT[],                       -- array of region strings
    scope            TEXT
        CHECK (scope IN ('national', 'regional')), -- restrict values
    startdate        DATE,                         -- when the event began

    /* business-key so the loader knows what “same row” means */
    CONSTRAINT explanations_summary_startdate_uk
        UNIQUE (summary, startdate),

    /* enforce: cause allowed ⇔ tags include 'censorship' */
    CONSTRAINT cause_requires_censorship_tag
        CHECK (
              cause IS NULL
           OR tags @> ARRAY['censorship']
        )
    
);

/* helpful indexes for “contains” queries */
CREATE INDEX explanations_links_gin   ON explanations USING GIN (links);
CREATE INDEX explanations_tags_gin    ON explanations USING GIN (tags);
CREATE INDEX explanations_regions_gin ON explanations USING GIN (affected_regions);

CREATE TABLE anomalies (
    country_code   TEXT           NOT NULL,
    start_date     DATE           NOT NULL,
    end_date       DATE           NOT NULL,
    score          DOUBLE PRECISION,
    residual       DOUBLE PRECISION,
    impact         DOUBLE PRECISION NOT NULL,
    terms          TEXT[]         NOT NULL DEFAULT '{}'::text[],
    explanation_id INTEGER        REFERENCES explanations(id) ON DELETE SET NULL,

    PRIMARY KEY (country_code, start_date),

    CHECK (
        (score = 0   AND residual IS NOT NULL) OR
        (residual = 0 AND score    IS NOT NULL)
    ),
    CHECK (start_date <= end_date)
);

