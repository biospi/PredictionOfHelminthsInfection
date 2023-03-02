from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

from utils.Utils import anscombe


# aliceblue, antiquewhite, aqua, aquamarine, azure,
#             beige, bisque, black, blanchedalmond, blue,
#             blueviolet, brown, burlywood, cadetblue,
#             chartreuse, chocolate, coral, cornflowerblue,
#             cornsilk, crimson, cyan, darkblue, darkcyan,
#             darkgoldenrod, darkgray, darkgrey, darkgreen,
#             darkkhaki, darkmagenta, darkolivegreen, darkorange,
#             darkorchid, darkred, darksalmon, darkseagreen,
#             darkslateblue, darkslategray, darkslategrey,
#             darkturquoise, darkviolet, deeppink, deepskyblue,
#             dimgray, dimgrey, dodgerblue, firebrick,
#             floralwhite, forestgreen, fuchsia, gainsboro,
#             ghostwhite, gold, goldenrod, gray, grey, green,
#             greenyellow, honeydew, hotpink, indianred, indigo,
#             ivory, khaki, lavender, lavenderblush, lawngreen,
#             lemonchiffon, lightblue, lightcoral, lightcyan,
#             lightgoldenrodyellow, lightgray, lightgrey,
#             lightgreen, lightpink, lightsalmon, lightseagreen,
#             lightskyblue, lightslategray, lightslategrey,
#             lightsteelblue, lightyellow, lime, limegreen,
#             linen, magenta, maroon, mediumaquamarine,
#             mediumblue, mediumorchid, mediumpurple,
#             mediumseagreen, mediumslateblue, mediumspringgreen,
#             mediumturquoise, mediumvioletred, midnightblue,
#             mintcream, mistyrose, moccasin, navajowhite, navy,
#             oldlace, olive, olivedrab, orange, orangered,
#             orchid, palegoldenrod, palegreen, paleturquoise,
#             palevioletred, papayawhip, peachpuff, peru, pink,
#             plum, powderblue, purple, red, rosybrown,
#             royalblue, rebeccapurple, saddlebrown, salmon,
#             sandybrown, seagreen, seashell, sienna, silver,
#             skyblue, slateblue, slategray, slategrey, snow,
#             springgreen, steelblue, tan, teal, thistle, tomato,
#             turquoise, violet, wheat, white, whitesmoke,
#             yellow, yellowgreen


#list of transponders sorted by entropy

# transponders_delmas = [
#     "40101310316",
#     "40101310040",
#     "40101310109",
#     "40101310110",
#     "40101310353",
#     "40101310314",
#     "40101310085",
#     "40101310143",
#     "40101310409",
#     "40101310134",
#     "40101310342",
#     "40101310069",
#     "40101310013",
#     "40101310098",
#     "40101310350",
#     "40101310386",
#     "40101310249",
#     "40101310121",
#     "40101310137",
#     "40101310241",
#     "40101310119",
#     "40101310081",
#     "40101310299",
#     "40101310094",
#     "40101310345",
#     "40101310220",
#     "40101310224",
#     "40101310231",
#     "40101310026",
#     "40101310395",
#     "40101310318",
#     "40101310039",
#     "40101310125",
#     "40101310247",
#     "40101310389",
#     "40101310228",
#     "40101310145",
#     "40101310238",
#     "40101310230",
#     "40101310142",
#     "40101310092",
#     "40101310336",
#     "40101310106",
#     "40101310347",
#     "40101310071",
#     "40101310083",
#     "40101310036",
#     "40101310332",
#     "40101310310",
#     "40101310157",
#     "40101310117",
#     "40101310115",
#     "40101310050",
#     "40101310095",
#     "40101310016",
#     "40101310239",
#     "40101310229",
#     "40101310086",
#     "40101310352",
#     "40101310100",
#     "40101310052",
#     "40101310135",
#     "40101310044",
#     "40101310107",
#     "40101310146",
#     "40101310123",
# ]

transponders_delmas = ['40101310316', '40101310314', '40101310109', '40101310143', '40101310353', '40101310409',
                       '40101310134', '40101310069', '40101310040', '40101310110', '40101310350', '40101310386',
                       '40101310249', '40101310121', '40101310013', '40101310085', '40101310342', '40101310098',
                       '40101310119', '40101310081', '40101310299', '40101310345', '40101310137', '40101310094',
                       '40101310241', '40101310224', '40101310220', '40101310039', '40101310231', '40101310026',
                       '40101310318', '40101310389', '40101310238', '40101310071', '40101310228', '40101310336',
                       '40101310036', '40101310050', '40101310092', '40101310145', '40101310247', '40101310395',
                       '40101310083', '40101310117', '40101310125', '40101310142', '40101310332', '40101310106',
                       '40101310310', '40101310347', '40101310230', '40101310095', '40101310239', '40101310086',
                       '40101310157', '40101310016', '40101310052', '40101310115', '40101310044', '40101310229',
                       '40101310100', '40101310352', '40101310107', '40101310135', '40101310146', '40101310123']

# transponders_cedara = [
#     "40011301509",
#     "40061201018",
#     "40061201134",
#     "40061200951",
#     "40061201015",
#     "40011301512",
#     "40011301539",
#     "40011301542",
#     "40011301557",
#     "40011301511",
#     "40011301510",
#     "40011301575",
#     "40011301599",
#     "40061201024",
#     "40011301556",
#     "40011301559",
#     "40011301596",
#     "40061201077",
#     "40011301527",
#     "40011301581",
#     "40011301565",
#     "40011301528",
#     "40011301546",
#     "40011301551",
#     "40061200966",
#     "40011301545",
#     "40061200930",
#     "40011301568",
#     "40061200880",
#     "40011301590",
#     "40061200862",
#     "40011301597",
#     "40011301520",
#     "40061200928",
#     "40061201042",
#     "40061200873",
#     "40061200929",
#     "40061201055",
#     "40061200934",
#     "40061200922",
#     "40061200910",
#     "40061200856",
#     "40011301521",
#     "40011301515",
#     "40011301589",
#     "40011301501",
#     "40061201073",
#     "40011301591",
#     "40061200845",
#     "40121100718",
#     "40011301573",
#     "40011301529",
#     "40061201006",
#     "40061201129",
#     "40061201068",
#     "40011301504",
#     "40011301585",
#     "40011301517",
#     "40061200947",
#     "40011301569",
#     "40011301534",
#     "40061200838",
#     "40011301577",
#     "40011301564",
#     "40061200944",
#     "40011301566",
#     "40011301524",
#     "40061201010",
#     "40011301553",
#     "40061200978",
#     "40061201037",
#     "40061200999",
#     "40061200904",
#     "40011301519",
#     "40011301580",
#     "40061201135",
#     "40011301595",
#     "40011301571",
#     "40011301508",
#     "40011301523",
#     "40121100797",
#     "40061200986",
#     "40061201091",
#     "40011301503",
#     "40011301548",
#     "40061201127",
#     "40061201021",
#     "40061200879",
#     "40061201115",
#     "40061201097",
#     "40011301537",
#     "40061200889",
#     "40061200837",
#     "40061200881",
#     "40061201035",
#     "40061201054",
#     "40061201123",
#     "40011301555",
#     "40061201028",
#     "40061201074",
#     "40011301600",
#     "40061200868",
#     "40061201049",
#     "40121100243",
#     "40121100582",
#     "40061201089",
#     "40061201098",
#     "40061200848",
#     "40061200896",
#     "40061200989",
#     "40061200836",
#     "40011301579",
#     "40061201084",
#     "40061201131",
#     "40121100868",
#     "40061201102",
#     "40011301560",
#     "40061201036",
#     "40011301532",
#     "40061200915",
#     "40011301767",
#     "40011301710",
#     "40011301507",
#     "40011301799",
#     "40011301707",
#     "40011301506",
#     "40011301728",
#     "40061200923",
#     "40011301593",
#     "40061200921",
#     "40011301720",
#     "40011301777",
#     "40061201012",
#     "40061200967",
#     "40061201095",
#     "40061200970",
#     "40061200965",
#     "40011301711",
#     "40011301765",
#     "40011301789",
#     "40011301790",
#     "40011301760",
#     "40061200908",
#     "40011301750",
#     "40011301744",
#     "40011301731",
#     "40011301782",
#     "40011301795",
#     "40061201105",
#     "40011301791",
#     "40011301732",
#     "40011301757",
#     "40011301730",
#     "40011301754",
#     "40011301797",
#     "40061201056",
#     "40011301753",
#     "40011301594",
#     "40011301751",
#     "40011301531"
#     # "40011301583",
#     # "40011301550"
#     # "40011301541",
#     # "40011301584",
#     # "40011301552",
#     # "40011301567",
#     # "40011301513",
#     # "40011301574",
#     # "40011301570",
#     # "40011301544",
#     # "40011301572",
#     # "40011301516",
# ]

transponders_cedara = ['40061201018', '40011301509', '40061200951', '40061201015', '40011301539', '40061201024',
                       '40061200966', '40011301542', '40061201077', '40011301575', '40061201134', '40011301512',
                       '40011301596', '40011301551', '40011301511', '40011301556', '40011301565', '40011301581',
                       '40011301557', '40061200930', '40061200880', '40011301559', '40011301510', '40011301527',
                       '40011301528', '40011301546', '40011301599', '40011301545', '40011301520', '40061201042',
                       '40011301568', '40011301590', '40061200856', '40061200928', '40121100718', '40061200873',
                       '40061200929', '40061201055', '40061201135', '40011301597', '40061200862', '40011301504',
                       '40121100797', '40061200910', '40011301501', '40011301521', '40011301573', '40061201006',
                       '40011301534', '40011301569', '40061200934', '40011301577', '40011301591', '40011301580',
                       '40061201129', '40011301564', '40011301566', '40011301519', '40061200944', '40061200922',
                       '40011301524', '40011301553', '40011301515', '40011301508', '40011301595', '40061200999',
                       '40011301523', '40061201010', '40011301517', '40061200838', '40061201037', '40011301571',
                       '40061201068', '40011301589', '40061201127', '40121100582', '40011301585', '40121100243',
                       '40011301529', '40061201073', '40011301503', '40061200845', '40061200947', '40061201091',
                       '40011301537', '40061200879', '40011301548', '40061200986', '40061200868', '40061200904',
                       '40011301555', '40061200896', '40061201115', '40061201089', '40061201074', '40061201097',
                       '40061201084', '40061200889', '40061200881', '40011301579', '40061201123', '40061201049',
                       '40061201102', '40011301600', '40061200837', '40061201021', '40061201131', '40061200989',
                       '40061200848', '40061201054', '40061200836', '40061201035', '40011301532', '40011301506',
                       '40061200967', '40061200978', '40061201036', '40121100868', '40011301799', '40061201098',
                       '40011301710', '40061201028', '40011301707', '40011301593', '40061201095', '40061200908',
                       '40011301767', '40011301560', '40061200915', '40011301777', '40061201105', '40011301750',
                       '40011301728', '40011301720', '40011301789', '40061200923', '40011301765', '40011301711',
                       '40011301790', '40011301760', '40011301731', '40011301507', '40061200970', '40011301782',
                       '40011301744', '40011301795', '40011301732', '40011301791', '40011301594', '40061200921',
                       '40011301757', '40011301797', '40011301730', '40011301754', '40061201012', '40061201056',
                       '40011301753', '40061200965', '40011301751', '40011301531', '40011301513', '40011301584',
                       '40011301567', '40011301550', '40011301583', '40011301552', '40011301574', '40011301572',
                       '40011301516', '40011301541', '40011301544', '40011301570']


marker = [
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "D",
    "d",
    "|",
    "_",
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
]

def main(transponders, farm, n_top, n_bottom, data_folder, resolution="7D"):
    tot = len(transponders)
    data_folder = Path(data_folder)
    files = [data_folder / f"{x}.csv" for x in transponders[0:n_top]]

    if n_bottom is not None:
        files = [data_folder / f"{x}.csv" for x in transponders[len(transponders) - n_bottom:len(transponders)]]

    ncol = 1
    nrow = len(files)
    fig, axes = plt.subplots(nrow, ncol, constrained_layout=True, sharex=True, figsize=(6.2, 5.0))
    all_transponders = []
    for i, file in enumerate(files):
        print(f"{i}/{len(files)}...")
        df = pd.read_csv(file)
        df["index"] = pd.to_datetime(df["date_str"], format="%Y-%m-%dT%H:%M")
        all_transponders.append(df)
        df = df.resample(resolution, on="index").mean()
        ax = axes[i]
        df[f"first_sensor_value {file.stem}"] = df['first_sensor_value']
        df.plot.bar(y=f"first_sensor_value {file.stem}", rot=-80, ax=ax)
        ax.set_xlabel(f"Time (resampled to {resolution})")
        ax.set_ylabel("Activity count")
        ax.set_xticklabels(df.index.format())
        interval = 2
        if "cedara" in farm.lower():
            interval = 3
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        #ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        # ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y/%m/%d"))
    fig.suptitle(f'Top {n_top}/{tot} transponders sorted by entropy (top to bottom) for {farm}')
    if n_bottom is not None:
        fig.suptitle(f'Bottom {n_top}/{tot} transponders sorted by entropy (top to bottom) for {farm}')
    fig.tight_layout()
    file_path = data_folder / f"{farm}_top_transponders.png".lower()
    if n_bottom is not None:
        file_path = data_folder / f"{farm}_bottom_transponders.png".lower()
    print(file_path)
    fig.savefig(str(file_path))
    plt.show()


def show_sorted_transponders(transponders, farm, data_folder):
    data_folder = Path(data_folder)
    files = [data_folder / f"{x}.csv" for x in transponders]
    df_all = pd.DataFrame()

    for i, file in enumerate(files):
        print(f"{i}/{len(files)}...")
        df = pd.read_csv(file)
        df_all[file.stem] = df["first_sensor_value"]
        xaxix_label = pd.to_datetime(df["date_str"], format="%Y-%m-%dT%H:%M")

    stride = 1440*7
    for i in range(0, df_all.shape[0], stride):
        start = i
        end = start + stride
        df_all_ = df_all.iloc[start:end, :]
        xaxix_label_ = xaxix_label[start:end]
        fig = go.Figure(data=go.Heatmap(
            z=np.log(anscombe(df_all_.values.T)),
            x=xaxix_label_,
            y=df_all_.columns,
            colorscale='Viridis'))

        fig.update_layout(
            title='Transponders sorted by entropy', xaxis_title="Time (1 min bins)", yaxis_title="Transponders")

        filename = f"{i}_{farm}_sorted_by_entropy.html"
        out_dir = data_folder / filename
        # data_folder.mkdir(parents=True, exist_ok=True)
        print(out_dir)
        fig.write_html(out_dir)


if __name__ == "__main__":
    show_sorted_transponders(transponders_delmas, "Delmas", 'E:/thesis/activity_data/delmas/backfill_1min_delmas_fixed')
    show_sorted_transponders(transponders_cedara, "Cedara", 'E:/thesis/activity_data/cedara/backfill_1min_cedara_fixed')
    n_top = 3
    main(transponders_delmas, "Delmas", n_top, None, 'E:/thesis/activity_data/delmas/backfill_1min_delmas_fixed')
    main(transponders_delmas, "Delmas", n_top, n_top, 'E:/thesis/activity_data/delmas/backfill_1min_delmas_fixed')
    main(transponders_cedara, "Cedara", n_top, None, 'E:/thesis/activity_data/cedara/backfill_1min_cedara_fixed')
    main(transponders_cedara, "Cedara", n_top, n_top, 'E:/thesis/activity_data/cedara/backfill_1min_cedara_fixed')






