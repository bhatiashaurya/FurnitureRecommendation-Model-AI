"""
GenAI Product Description Generator using LangChain
Generates creative and compelling product descriptions
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

load_dotenv()

class ProductDescriptionGenerator:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the GenAI description generator
        
        Args:
            model_name: OpenAI model to use (gpt-3.5-turbo, gpt-4, etc.)
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize LangChain LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            openai_api_key=self.api_key
        )
        
        # Create prompt template for product descriptions
        self.description_template = PromptTemplate(
            input_variables=["title", "brand", "category", "material", "color", "price", "features"],
            template="""You are a creative furniture product description writer. Generate an engaging and persuasive product description based on the following details:

Product Title: {title}
Brand: {brand}
Category: {category}
Material: {material}
Color: {color}
Price: {price}
Key Features: {features}

Write a compelling 2-3 sentence product description that:
1. Highlights the product's unique features and benefits
2. Uses descriptive and appealing language
3. Appeals to the target customer's lifestyle and needs
4. Maintains a professional yet friendly tone

Product Description:"""
        )
        
        # Create LangChain chain
        self.description_chain = LLMChain(
            llm=self.llm,
            prompt=self.description_template
        )
    
    def generate_description(self, product_data: Dict) -> str:
        """
        Generate a creative product description
        
        Args:
            product_data: Dictionary containing product information
            
        Returns:
            Generated description string
        """
        try:
            # Extract product features
            title = product_data.get("title", "Furniture Item")
            brand = product_data.get("brand", "Unknown Brand")
            category = product_data.get("categories", ["Furniture"])[0] if product_data.get("categories") else "Furniture"
            material = product_data.get("material", "Quality Materials")
            color = product_data.get("color", "Versatile Color")
            price = product_data.get("price", "Affordable")
            
            # Extract features from existing description or use default
            existing_desc = product_data.get("description", "")
            features = existing_desc[:200] if existing_desc else "Stylish and functional design"
            
            # Generate description
            description = self.description_chain.run(
                title=title,
                brand=brand,
                category=category,
                material=material,
                color=color,
                price=price,
                features=features
            )
            
            return description.strip()
            
        except Exception as e:
            print(f"Error generating description: {str(e)}")
            return product_data.get("description", "Quality furniture piece for your home.")
    
    def generate_batch_descriptions(self, products: List[Dict]) -> List[str]:
        """
        Generate descriptions for multiple products
        
        Args:
            products: List of product dictionaries
            
        Returns:
            List of generated descriptions
        """
        descriptions = []
        
        for product in products:
            desc = self.generate_description(product)
            descriptions.append(desc)
        
        return descriptions
    
    def generate_recommendation_summary(self, query: str, products: List[Dict]) -> str:
        """
        Generate a summary for recommended products
        
        Args:
            query: User's search query
            products: List of recommended products
            
        Returns:
            Summary text
        """
        summary_template = PromptTemplate(
            input_variables=["query", "num_products", "product_titles"],
            template="""Based on the search query: "{query}"

I found {num_products} furniture items that match your needs:

{product_titles}

Generate a friendly 1-2 sentence introduction to these recommendations that:
1. Acknowledges the user's search intent
2. Briefly highlights what makes these products suitable
3. Encourages exploration

Introduction:"""
        )
        
        summary_chain = LLMChain(llm=self.llm, prompt=summary_template)
        
        product_titles = "\n".join([f"- {p.get('title', 'Product')}" for p in products])
        
        try:
            summary = summary_chain.run(
                query=query,
                num_products=len(products),
                product_titles=product_titles
            )
            return summary.strip()
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return f"Here are {len(products)} products that match '{query}'"


# Example usage
if __name__ == "__main__":
    generator = ProductDescriptionGenerator()
    
    # Example product
    sample_product = {
        "title": "Modern Leather Dining Chair",
        "brand": "StyleHome",
        "categories": ["Dining Room Furniture", "Chairs"],
        "material": "Leather",
        "color": "Black",
        "price": "$89.99",
        "description": "Comfortable dining chair with ergonomic design and sturdy construction"
    }
    
    # Generate description
    generated_desc = generator.generate_description(sample_product)
    print("Generated Description:")
    print(generated_desc)
    
    # Generate summary
    sample_products = [sample_product] * 3
    summary = generator.generate_recommendation_summary("modern dining chairs", sample_products)
    print("\nGenerated Summary:")
    print(summary)
