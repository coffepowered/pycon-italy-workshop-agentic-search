# %%
import lancedb

uri = "tmp.db"
db = lancedb.connect(uri)

table = db.create_table(
    "pycon_demo",
    data=[
        # 1. The Baseline
        {"text": "I am attending a great talk about Python at PyCon Italy in Bologna.", "vector":[0,0]},
        # 2. IDF 
        {"text": "Bologna is famous for ragù, tortellini, and amazing food.", "vector":[0,0]},
        # 3. TF saturation
        {"text": "Python is a language. Python code is clean. Python handles data. Python, Python, Python!", "vector":[0,0]},
        # 4. Length Normalization Showcase (Short doc vs. Long doc penalty)
        {"text": "Python at PyCon.", "vector":[0,0]},
        {"text": "If you are traveling to Italy for a conference, you should know that Python is widely used by engineers attending PyCon in the city of Bologna, which is hosting the event this year among many other interesting local tech meetups.", "vector":[0,0]},
    ],
    mode="overwrite"
)
# %%
table.create_fts_index("text")

# %%
# Why the first mathch?
results = table.search("Bologna ").limit(10).select(["text","_score"]).to_list()
results

# %%
# Which Python?
results = table.search("Python ").limit(10).select(["text","_score"]).to_list()
results
# %%
# What's text lenght normalization doing here?
results = table.search("Conference Italy ").limit(10).select(["text","_score"]).to_list()
results

# %%
