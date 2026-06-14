#
# EDA shiny app
#
# data size: 1024MB
options(shiny.maxRequestSize = 1024*1024^2)

library(shiny)
library(colourpicker)
library(shinyTime)
library(shinyWidgets)
library(bs4Dash)
library(DataExplorer)
library(gt)

Missing.Table <- function(data){
  output <- data.frame(feature= names(data),
                      type= sapply(data,class),
                      unique_value= sapply(data, function(x){if(!is.numeric(x)){length(unique(x))}
                        else{"---"}}),
                      num_missing= sapply(data, function(x){sum(is.na(x))}),
                      missing_rate= sapply(data, function(x) round(mean(is.na(x))*100, 2)))
  colnames(output) <-(c("feature","type","#ofuniquevalue",
                        "#ofmissingvalue", "missingrate(%)"))
  rownames(output) <-NULL
  return(output)
}


# Define UI for application that draws a histogram
ui <- bs4DashPage(
  titlePanel("Exploratory Data Analysis"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "上傳 CSV 檔案"),
      uiOutput("dynamic_input")
    ),
    
    mainPanel(
      tabsetPanel(id = "tabs",
                  tabPanel("App Introduction",verbatimTextOutput("intro")),
                  tabPanel("Data Preview", 
                           textOutput("dim"),
                           gt_output("head"), 
                           gt_output("missing"),
                           gt_output("summary")),
                  tabPanel("Data plot",plotOutput("hist"),plotOutput("scatter"))
      )
      
    )
  )
)

# Define server logic required to draw a histogram
server <- function(input, output){
  data <- reactive({
    req(input$file)
    read.csv(input$file$datapath)
  })

  output$dim <- renderText(dim(data())) 
  output$head <- render_gt({
    head(data()) %>% gt()
  })
  output$missing <- render_gt({
    Missing.Table(data())%>% gt()
  })
  output$summary <- render_gt({
    summary(data())
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
