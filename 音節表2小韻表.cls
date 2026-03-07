VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "音節表"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = True
Option Explicit

' 获取指定颜色的文本
Function GetColorText(pRange As Range, textColor As Long) As String
    Dim outputText As String
    Dim cellText As String
    Dim i As Long
    
    cellText = pRange.Text
    For i = 1 To VBA.Len(cellText)
        If pRange.Characters(i, 1).Font.Color = textColor Then
            outputText = outputText & VBA.Mid(cellText, i, 1)
        End If
    Next i
    
    GetColorText = outputText
End Function

' 提取单元格中所有颜色的文本
Function GetAllColoredText(pRange As Range) As Collection
    Dim result As New Collection
    Dim colorText As String
    
    ' 检查黑色文本
    colorText = GetColorText(pRange, 0)
    If Len(colorText) > 0 Then
        result.Add Array(colorText, 0)
    End If
        
    ' 检查蓝色文本
    colorText = GetColorText(pRange, RGB(0, 112, 192))
    If Len(colorText) > 0 Then
        result.Add Array(colorText, RGB(0, 112, 192))
    End If
    
    ' 检查红色文本
    colorText = GetColorText(pRange, RGB(255, 0, 0))
    If Len(colorText) > 0 Then
        result.Add Array(colorText, RGB(255, 0, 0))
    End If
    
    ' 检查灰色文本
    colorText = GetColorText(pRange, RGB(128, 128, 128))
    If Len(colorText) > 0 Then
        result.Add Array(colorText, RGB(128, 128, 128))
    End If
    
    Set GetAllColoredText = result
End Function

Sub ExtractColoredTextAndReorganize()
    Dim ws As Worksheet
    Dim newWs As Worksheet
    Dim cell As Range
    Dim rowNum As Long
    Dim i As Long, j As Long
    Dim headerCol As String
    Dim headerRow As String
    Dim colorCollection As Collection
    Dim colorItem As Variant
    
    ' 设置工作表
    Set ws = ActiveSheet
    
    ' 创建新工作表（如果已存在则先删除）
    On Error Resume Next
    Application.DisplayAlerts = False
    ThisWorkbook.Sheets("ColorExtract").Delete
    Application.DisplayAlerts = True
    On Error GoTo 0
    
    Set newWs = ThisWorkbook.Sheets.Add(After:=ws)
    newWs.Name = "ColorExtract"
    
    ' 禁用屏幕更新以提高性能
    Application.ScreenUpdating = False
    
    ' 初始化新工作表的行计数器
    rowNum = 1
    
    ' 遍历每一行
        For i = 4 To ws.UsedRange.Rows.Count ' 从第4行开始（跳过3行表头）
        ' 遍历每一列
        For j = 5 To ws.UsedRange.Columns.Count
            Set cell = ws.Cells(i, j)
            
            ' 获取表头信息
            headerCol = ws.Cells(3, j).Text
            headerRow = ws.Cells(i, 4).Text
            ' 获取所有颜色的文本
            Set colorCollection = GetAllColoredText(cell)
            
            
            ' 处理非第1列的单元格
            If Not IsEmpty(cell) Then
                
                ' 处理每个颜色文字组合
                For Each colorItem In colorCollection
                    ' 在新工作表中写入数据
                    newWs.Cells(rowNum, 2).Value = colorItem(0)    ' 文字内容
                    ' 定义一个颜色标识字符变量
                    Dim colorIdentifier As String
                    
                    ' 根据颜色值设置标识符
                    Select Case colorItem(1)
                        Case RGB(255, 0, 0)
                            colorIdentifier = "h"
                        Case RGB(0, 112, 192)
                            colorIdentifier = ChrW(&H294)
                        Case RGB(128, 128, 128)
                            colorIdentifier = "k"
                        Case Else
                            colorIdentifier = ""
                    End Select
                    
                    ' 写入带颜色标识的表头组合
                    newWs.Cells(rowNum, 1).Value = headerCol & headerRow & colorIdentifier
                    
                    ' 设置颜色
                    newWs.Cells(rowNum, 2).Font.Color = colorItem(1)
                    
                    rowNum = rowNum + 1
                Next colorItem
            End If
        Next j
    Next i
    
    ' 调整列宽
    newWs.Columns("A:B").AutoFit
    
    ' 恢复屏幕更新
    Application.ScreenUpdating = True
    
    MsgBox "处理完成！结果在新工作表'ColorExtract'中。"
End Sub






