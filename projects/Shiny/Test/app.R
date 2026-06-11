library(shiny)
library(colourpicker)
library(shinyTime)
library(shinyWidgets)
# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel("Old Faithful Geyser Data"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      textInput("x1", "Fill in your name", value = 0, width = "200%",
                placeholder = "Kevin", updateOn = "blur"),
      passwordInput("x2", "Password"),
      textAreaInput("x3", "Fill in your story"),
      
      fileInput("data", "Select file"),
      
      selectInput("x4", "Hi", c("Male", "Female")),
      selectizeInput("x5", "Hi2", c("Male", "Female")),
      radioButtons("x6", "Hi3", c("True", "False")),
      checkboxInput("x7", "Hi4", value = TRUE),
      checkboxGroupInput("x8", "Hi5", c("A", "B", "C", "D")),
      pickerInput("x15", "pickerInput", c(1,2,3,4,5), multiple = T,
                  options = list(pickerOptions(liveSearch = T))),
      
      actionButton("x9", "actionButton"),
      actionBttn("x9.1", "actionButton-1"),
      actionLink("x10", "actionLink"),
      br(),br(),
      dateInput("x11", "dateInput:", value = "2012-02-29", format = "mm/dd/yy"),
      dateRangeInput("daterange3", "Date range:", start= "2001-01-01", 
                     end = "2010-12-31", min = "2001-01-01", max = "2012-12-21",
                     format = "mm/dd/yy",separator = " - "),
      
      numericInput("x12", "numericInput", value = 0),
      sliderInput("bins","Number of bins:",min = 1,max = 50, value = 30),
      
      colourInput("x13", "Choose color"),
      
      timeInput("x14", "Choose time")
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      plotOutput("distPlot"),
      textOutput("string")
    )
  )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  
  output$distPlot <- renderPlot({
    # generate bins based on input$bins from ui.R
    x    <- faithful[, 2]
    bins <- seq(min(x), max(x), length.out = input$bins + 1)
    
    # draw the histogram with the specified number of bins
    hist(x, breaks = bins, col = 'darkgray', border = 'white',
         xlab = 'Waiting time to next eruption (in mins)',
         main = 'Histogram of waiting times')
  })
  output$string <- renderText({
    paste(input$x15, collapse = "+")
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
