





#写Excel
def write_excel():

    f = xlwt.Workbook()
    sheet1 = f.add_sheet('stitch',cell_overwrite_ok=True)
    row0 = ["姓名","年龄","出生日期","爱好"]
    colum0 = ["张三","李四","恋习Python","小明","小红","无名"]
    #写第一行



    #写第一列




    sheet1.write(1,3,'2006/12/12')

    sheet1.write_merge(6,6,1,3,'未知')#合并行单元格

    sheet1.write_merge(1,2,3,3,'打游戏')#合并列单元格

    sheet1.write_merge(4,5,3,3,'打篮球')


    f.save('/media/liuxz/3EA0B4CEA0B48E41/mb_aligner/my_mb_aligner/output/test.xls')


if __name__ == '__main__':
    write_excel()